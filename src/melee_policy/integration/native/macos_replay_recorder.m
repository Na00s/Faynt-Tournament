#import <AVFoundation/AVFoundation.h>
#import <AppKit/AppKit.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CommonCrypto/CommonDigest.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

// Half-speed gameplay changes at roughly 30 Hz. Allow one 33.3 ms state
// interval plus compositor jitter while rejecting the roughly 66.7 ms gap
// produced when an entire state delivery is absent.
static const double MaximumDeliveryGapSeconds = 0.040;
// A ScreenCaptureKit callback can straddle the one controlled terminal resume
// and SIGSTOP handoff. Keep that narrowly recoverable window below 50 ms. The
// original callback gap remains in the trace and is admitted only after the
// terminal barrier-through-stopped-hold proof has been established.
static const double TerminalDeliveryGapRecoveryCeilingSeconds = 0.050;
static const double MaximumCallbackDeliveryLagSeconds = 0.050;
static const double MaximumCallbackServiceSeconds = 1.0 / 120.0;
// The immutable cutoff is the final callback of the separately proven pre-roll.
// The first pair crossing it and every terminal callback remain strict.
static BOOL strictCapturePair(NSUInteger currentSequence, NSUInteger startupCallbackCount) {
  return currentSequence > startupCallbackCount;
}
static const double MaximumSustainedBlankSeconds = 0.500;
static const double RequiredBlankPixelFraction = 0.995;
static const uint8_t NearWhiteVideoLumaMinimum = 226;
static const uint8_t NearBlackVideoLumaMaximum = 25;
static const double PlaybackLogQuietSeconds = 0.050;
static const double EncodedPTSSequenceErrorSeconds = 1.0 / 60000.0 + 0.000001;
static const double EncodedPTSTraceRoundingSeconds = 0.000001;
static const double EncodedPTSNumericalSlackSeconds = 0.000000000001;
// A terminal CURRENT_FRAME stop can expose one final emulated-frame write plus
// at most 2 ms of WaveFile scheduling and sample-phase skew. The deterministic content seal
// later trims this physical payload to the normalized sound-sync endpoint.
static const double TerminalAudioMaximumPhysicalOvershootSeconds =
    1.0 / 60.0 + 0.002000;
static const double TerminalAudioFinalizedBufferAllowanceSeconds = 0.100;
static const double TerminalAudioStopStabilityIntervalSeconds = 0.025;
static const double TerminalApplicationProofMaximumSeconds = 1.000;

static double encodedPTSGapReconciliationErrorSeconds(void) {
  return 2.0 * EncodedPTSSequenceErrorSeconds + EncodedPTSTraceRoundingSeconds;
}

static double maximumEncodedPTSGapSeconds(void) {
  return MaximumDeliveryGapSeconds + encodedPTSGapReconciliationErrorSeconds();
}

static NSNumber *jsonBoolean(BOOL value) {
  return [NSNumber numberWithBool:value];
}

static NSString *sha256Hex(NSData *data) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
  NSMutableString *result = [NSMutableString stringWithCapacity:CC_SHA256_DIGEST_LENGTH * 2];
  for (NSUInteger index = 0; index < CC_SHA256_DIGEST_LENGTH; ++index)
    [result appendFormat:@"%02x", digest[index]];
  return result;
}

typedef NS_ENUM(NSInteger, CaptureFrameClass) {
  CaptureFrameClassActive,
  CaptureFrameClassNearWhite,
  CaptureFrameClassNearBlack,
  CaptureFrameClassNeutralBlank,
  CaptureFrameClassUnknown,
};

typedef struct {
  CaptureFrameClass frameClass;
  uint64_t signature;
} CaptureFrameClassification;

enum {
  TraceFlagValidSample = 1 << 0,
  TraceFlagValidStatus = 1 << 1,
  TraceFlagValidDisplayTime = 1 << 2,
  TraceFlagDropAttachment = 1 << 3,
  TraceFlagValidPixel = 1 << 4,
  TraceFlagAppended = 1 << 5,
  TraceFlagIdleHold = 1 << 6,
  TraceFlagStopped = 1 << 7,
  TraceFlagGameplayArmed = 1 << 8,
  TraceFlagActiveContent = 1 << 9,
  TraceFlagStopRequested = 1 << 10,
};

static NSError *captureError(NSString *message) {
  return [NSError errorWithDomain:@"melee-policy.replay-recorder"
                             code:1
                         userInfo:@{NSLocalizedDescriptionKey: message}];
}

static NSDictionary *firstSampleAttachment(CMSampleBufferRef sampleBuffer) {
  CFArrayRef attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, false);
  if (!attachments || CFArrayGetCount(attachments) < 1) return nil;
  CFTypeRef value = CFArrayGetValueAtIndex(attachments, 0);
  if (!value || CFGetTypeID(value) != CFDictionaryGetTypeID()) return nil;
  return (__bridge NSDictionary *)value;
}

static NSString *frameStatusName(SCFrameStatus status) {
  switch (status) {
    case SCFrameStatusComplete: return @"complete";
    case SCFrameStatusIdle: return @"idle";
    case SCFrameStatusBlank: return @"blank";
    case SCFrameStatusSuspended: return @"suspended";
    case SCFrameStatusStarted: return @"started";
    case SCFrameStatusStopped: return @"stopped";
  }
  return @"unknown";
}

static char frameStatusCode(SCFrameStatus status) {
  switch (status) {
    case SCFrameStatusComplete: return 'C';
    case SCFrameStatusIdle: return 'I';
    case SCFrameStatusBlank: return 'B';
    case SCFrameStatusSuspended: return 'U';
    case SCFrameStatusStarted: return 'S';
    case SCFrameStatusStopped: return 'T';
  }
  return 'X';
}

static char frameClassCode(CaptureFrameClass frameClass) {
  switch (frameClass) {
    case CaptureFrameClassActive: return 'A';
    case CaptureFrameClassNearWhite: return 'W';
    case CaptureFrameClassNearBlack: return 'K';
    case CaptureFrameClassNeutralBlank: return 'N';
    case CaptureFrameClassUnknown: return 'X';
  }
  return 'X';
}

static CaptureFrameClassification classifyCentralGameplayPixels(CVPixelBufferRef pixelBuffer) {
  CaptureFrameClassification result = {CaptureFrameClassUnknown, 0};
  if (!pixelBuffer || !CVPixelBufferIsPlanar(pixelBuffer) ||
      CVPixelBufferGetPlaneCount(pixelBuffer) < 1)
    return result;
  if (CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess)
    return result;
  uint8_t *base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);
  size_t width = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0);
  size_t height = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0);
  size_t stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0);
  NSUInteger samples = 0;
  NSUInteger nearWhite = 0;
  NSUInteger nearBlack = 0;
  uint64_t signature = UINT64_C(1469598103934665603);
  if (base && width >= 8 && height >= 6 && stride >= width) {
    size_t left = width / 8;
    size_t right = width - width / 8;
    size_t top = height / 6;
    size_t bottom = height - height / 6;
    for (size_t y = top; y < bottom; y += 4) {
      const uint8_t *row = base + y * stride;
      for (size_t x = left; x < right; x += 4) {
        uint8_t luma = row[x];
        signature ^= luma;
        signature *= UINT64_C(1099511628211);
        nearWhite += luma >= NearWhiteVideoLumaMinimum;
        nearBlack += luma <= NearBlackVideoLumaMaximum;
        samples += 1;
      }
    }
  }
  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  if (samples == 0) return result;
  result.signature = signature;
  double whiteFraction = (double)nearWhite / (double)samples;
  double blackFraction = (double)nearBlack / (double)samples;
  double neutralFraction = (double)(nearWhite + nearBlack) / (double)samples;
  if (whiteFraction >= RequiredBlankPixelFraction)
    result.frameClass = CaptureFrameClassNearWhite;
  else if (blackFraction >= RequiredBlankPixelFraction)
    result.frameClass = CaptureFrameClassNearBlack;
  else if (neutralFraction >= RequiredBlankPixelFraction)
    result.frameClass = CaptureFrameClassNeutralBlank;
  else
    result.frameClass = CaptureFrameClassActive;
  return result;
}

@interface CaptureWriter : NSObject <SCStreamDelegate, SCStreamOutput>
@property(nonatomic) dispatch_queue_t sampleQueue;
@property(nonatomic) dispatch_semaphore_t firstFrameReady;
@property(nonatomic) dispatch_semaphore_t writerFinished;
@property(nonatomic, strong) NSError *failure;
@property(nonatomic, strong) AVAssetWriter *writer;
@property(nonatomic, strong) AVAssetWriterInput *videoInput;
@property(nonatomic, strong) AVAssetWriterInputPixelBufferAdaptor *adaptor;
@property(nonatomic) CVPixelBufferRef lastPixelBuffer;
@property(nonatomic) BOOL sessionStarted;
@property(nonatomic) BOOL startupSignaled;
@property(nonatomic) BOOL stopRequested;
@property(nonatomic) BOOL stopCompletionObserved;
@property(nonatomic) BOOL stopCompletionSucceeded;
@property(nonatomic) BOOL stopCompletionFollowedRequest;
@property(nonatomic) BOOL stoppedStatusObserved;
@property(nonatomic) BOOL stoppedStatusFollowedRequest;
@property(nonatomic) BOOL delegateStopCallbackObserved;
@property(nonatomic) BOOL delegateStopErrorObserved;
@property(nonatomic) BOOL acceptingSamples;
@property(nonatomic) CMTime sessionStartHostPTS;
@property(nonatomic) CMTime lastAppendedHostPTS;
@property(nonatomic) CMTime stopHostPTS;
@property(nonatomic) uint64_t firstDisplayTime;
@property(nonatomic) uint64_t lastDisplayTime;
@property(nonatomic) uint64_t lastCompleteDisplayTime;
@property(nonatomic) NSUInteger lastCompleteCallbackSequence;
@property(nonatomic) NSUInteger callbackCount;
@property(nonatomic) NSUInteger appendedStartedCount;
@property(nonatomic) NSUInteger appendedCompleteCount;
@property(nonatomic) NSUInteger appendedIdleCount;
@property(nonatomic) NSUInteger appendedTerminalRecoveryCount;
@property(nonatomic) NSUInteger terminalHoldCount;
@property(nonatomic) NSUInteger appendedSampleCount;
@property(nonatomic) NSUInteger dropAttachmentCount;
@property(nonatomic) NSUInteger writerBackpressureCount;
@property(nonatomic) NSUInteger appendFailureCount;
@property(nonatomic) NSUInteger invalidSampleCount;
@property(nonatomic) double maximumDisplayGapSeconds;
@property(nonatomic) double maximumCallbackDeliveryLagSeconds;
@property(nonatomic) double maximumCallbackServiceSeconds;
@property(nonatomic) double terminalCoverageGapSeconds;
@property(nonatomic) BOOL gameplayArmed;
@property(nonatomic) BOOL startupPhaseComplete;
@property(nonatomic) NSUInteger startupCallbackCount;
@property(nonatomic) double startupSourceVideoSeconds;
@property(nonatomic) BOOL terminalGapRecoveryArmed;
@property(nonatomic) BOOL finalFrozenTailSealed;
@property(nonatomic) uint64_t finalFrozenTailVisualSignature;
@property(nonatomic) CaptureFrameClass finalFrozenTailContentClass;
@property(nonatomic) CaptureFrameClass lastFrameClass;
@property(nonatomic) CaptureFrameClass blankRunClass;
@property(nonatomic) CMTime blankRunStartHostPTS;
@property(nonatomic) double longestNearWhiteSeconds;
@property(nonatomic) double longestNearBlackSeconds;
@property(nonatomic) double longestNeutralBlankSeconds;
@property(nonatomic) NSUInteger classifiedFrameCount;
@property(nonatomic) NSUInteger visualSignatureTransitionCount;
@property(nonatomic) NSUInteger contentClassTransitionCount;
@property(nonatomic) BOOL appendedVisualStateInitialized;
@property(nonatomic) uint64_t lastAppendedVisualSignature;
@property(nonatomic) CaptureFrameClass lastAppendedContentClass;
@property(nonatomic, strong) NSMutableString *callbackTraceRows;
@property(nonatomic, strong) NSMutableArray<NSMutableArray *> *callbackTrace;
@property(nonatomic) uint64_t lastVisualSignature;
@property(nonatomic) int requestedCaptureFPS;
@property(nonatomic) size_t width;
@property(nonatomic) size_t height;
@property(nonatomic, strong) NSMutableDictionary<NSString *, NSNumber *> *statusCounts;
@property(nonatomic, strong) NSMutableDictionary<NSString *, NSNumber *> *dropReasons;
@property(nonatomic, strong) NSMutableArray<NSNumber *> *appendedRelativePTSSeconds;
@property(nonatomic, strong) NSMutableString *appendedPTSTraceRows;
@property(nonatomic, strong) NSMutableArray<NSDictionary *> *terminalGapRecoveries;
@property(nonatomic, strong) NSMutableString *terminalGapRecoveryRows;
@property(nonatomic, strong) NSDictionary *terminalGapRecoveryMetadata;
- (instancetype)initWithURL:(NSURL *)url width:(size_t)width height:(size_t)height
        requestedCaptureFPS:(int)requestedCaptureFPS;
- (NSError *)failureSnapshot;
- (double)currentSourceVideoSeconds;
- (NSDictionary *)boundarySnapshot;
- (NSDictionary *)boundarySnapshotAndArmTerminalGapRecovery;
- (NSError *)validateTerminalGapRecoveriesFromBarrier:(NSDictionary *)barrier
                                      acceptedComplete:(NSDictionary *)accepted
                                    throughFinalFrozenTail:(NSDictionary *)finalFrozenTail;
- (NSDictionary *)validateFinalFrozenTailFromAccepted:(NSDictionary *)accepted
                                        throughBoundary:(NSDictionary *)finalBoundary;
- (NSDictionary *)validateSealedStopSuffixFromFinalBoundary:(NSDictionary *)finalBoundary
                                             acceptedComplete:(NSDictionary *)accepted;
- (BOOL)waitForCallbackAfterSequence:(NSUInteger)sequence
                             timeout:(double)timeoutSeconds;
- (NSDictionary *)armGameplayAtHostTime:(CMTime)hostPTS;
- (NSDictionary *)disarmGameplayReturningBoundary;
- (void)prepareToStopAtHostTime:(CMTime)stopHostPTS;
- (void)recordStopCompletionWithError:(NSError *)error;
- (BOOL)insertTerminalGapRecoveryAtHostTime:(CMTime)hostPTS
                    visualSignature:(uint64_t)visualSignature
                        contentClass:(CaptureFrameClass)contentClass;
- (BOOL)commitTerminalGapRecoveryEvidence:(NSDictionary *)recovery;
- (void)finishWritingAtHostTime:(CMTime)stopHostPTS;
- (NSArray<NSNumber *> *)appendedPTSSequenceSnapshot;
- (NSDictionary *)deliveryMetadataWithPostWriteAudit:(NSDictionary *)postWriteAudit;
@end

@implementation CaptureWriter

- (instancetype)initWithURL:(NSURL *)url width:(size_t)width height:(size_t)height
        requestedCaptureFPS:(int)requestedCaptureFPS {
  if ((self = [super init])) {
    _sampleQueue = dispatch_queue_create("melee-policy.replay-recorder.samples",
                                         DISPATCH_QUEUE_SERIAL);
    _firstFrameReady = dispatch_semaphore_create(0);
    _writerFinished = dispatch_semaphore_create(0);
    _acceptingSamples = YES;
    _sessionStartHostPTS = kCMTimeInvalid;
    _lastAppendedHostPTS = kCMTimeInvalid;
    _stopHostPTS = kCMTimeInvalid;
    _blankRunStartHostPTS = kCMTimeInvalid;
    _lastFrameClass = CaptureFrameClassUnknown;
    _finalFrozenTailContentClass = CaptureFrameClassUnknown;
    _blankRunClass = CaptureFrameClassUnknown;
    _requestedCaptureFPS = requestedCaptureFPS;
    _width = width;
    _height = height;
    _statusCounts = [@{
      @"started": @0, @"complete": @0, @"idle": @0, @"blank": @0,
      @"suspended": @0, @"stopped": @0, @"unknown": @0,
    } mutableCopy];
    _dropReasons = [NSMutableDictionary dictionary];
    _callbackTraceRows = [NSMutableString string];
    _callbackTrace = [NSMutableArray array];
    _appendedRelativePTSSeconds = [NSMutableArray array];
    _appendedPTSTraceRows = [NSMutableString string];
    _terminalGapRecoveries = [NSMutableArray array];
    _terminalGapRecoveryRows = [NSMutableString string];

    [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
    NSError *writerError = nil;
    _writer = [AVAssetWriter assetWriterWithURL:url
                                       fileType:AVFileTypeMPEG4
                                          error:&writerError];
    if (!_writer) {
      _failure = writerError ?: captureError(@"could not create AVAssetWriter");
      return self;
    }
    NSDictionary *compression = @{
      AVVideoExpectedSourceFrameRateKey: @(requestedCaptureFPS),
      AVVideoAverageNonDroppableFrameRateKey: @(requestedCaptureFPS),
      AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
      AVVideoAllowFrameReorderingKey: @NO,
      AVVideoMaxKeyFrameIntervalKey: @(requestedCaptureFPS),
      AVVideoAverageBitRateKey: @(30 * 1000 * 1000),
    };
    NSDictionary *settings = @{
      AVVideoCodecKey: AVVideoCodecTypeH264,
      AVVideoWidthKey: @(width),
      AVVideoHeightKey: @(height),
      AVVideoCompressionPropertiesKey: compression,
    };
    _videoInput = [AVAssetWriterInput assetWriterInputWithMediaType:AVMediaTypeVideo
                                                     outputSettings:settings];
    _videoInput.expectsMediaDataInRealTime = YES;
    _videoInput.mediaTimeScale = 60000;
    NSDictionary *pixelAttributes = @{
      (id)kCVPixelBufferPixelFormatTypeKey:
          @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
      (id)kCVPixelBufferWidthKey: @(width),
      (id)kCVPixelBufferHeightKey: @(height),
    };
    _adaptor = [AVAssetWriterInputPixelBufferAdaptor
        assetWriterInputPixelBufferAdaptorWithAssetWriterInput:_videoInput
                                   sourcePixelBufferAttributes:pixelAttributes];
    if (![_writer canAddInput:_videoInput]) {
      _failure = captureError(@"AVAssetWriter rejected the isolated-window video input");
      return self;
    }
    [_writer addInput:_videoInput];
    if (![_writer startWriting]) {
      _failure = _writer.error ?: captureError(@"could not start AVAssetWriter");
      return self;
    }
  }
  return self;
}

- (void)dealloc {
  if (_lastPixelBuffer) CVPixelBufferRelease(_lastPixelBuffer);
}

- (void)signalStartupIfNeeded {
  if (self.startupSignaled) return;
  self.startupSignaled = YES;
  dispatch_semaphore_signal(self.firstFrameReady);
}

- (void)setFailureOnSampleQueue:(NSError *)error {
  if (!self.failure) {
    self.failure = error;
    NSString *trace = self.callbackTraceRows ?: @"";
    NSString *tail = trace.length > 8192 ? [trace substringFromIndex:trace.length - 8192] : trace;
    NSDictionary *diagnostic = @{
      @"schema": @"sck-capture-failure-v1",
      @"error": error.localizedDescription ?: @"unknown",
      @"startup_phase_complete": @(self.startupPhaseComplete),
      @"startup_callback_count": @(self.startupCallbackCount),
      @"callback_count": @(self.callbackCount),
      @"last_display_time": @(self.lastDisplayTime),
      @"maximum_display_gap_seconds": @(self.maximumDisplayGapSeconds),
      @"callback_trace_tail": tail,
    };
    NSData *encoded = [NSJSONSerialization dataWithJSONObject:diagnostic options:0 error:nil];
    if (encoded) fprintf(stderr, "CAPTURE_FAILURE_JSON %.*s\n", (int)encoded.length,
                         (const char *)encoded.bytes);
  }
  self.acceptingSamples = NO;
  [self signalStartupIfNeeded];
}

- (void)incrementStatus:(NSString *)name {
  self.statusCounts[name] = @([self.statusCounts[name] unsignedIntegerValue] + 1);
}

- (void)replaceLastPixelBuffer:(CVPixelBufferRef)pixelBuffer {
  CVPixelBufferRetain(pixelBuffer);
  if (self.lastPixelBuffer) CVPixelBufferRelease(self.lastPixelBuffer);
  self.lastPixelBuffer = pixelBuffer;
}

- (BOOL)appendPixelBuffer:(CVPixelBufferRef)pixelBuffer
               atHostTime:(CMTime)hostPTS
                      kind:(NSString *)kind {
  if (!self.videoInput.readyForMoreMediaData) {
    self.writerBackpressureCount += 1;
    [self setFailureOnSampleQueue:captureError(@"AVAssetWriter applied video backpressure")];
    return NO;
  }
  if (![self.adaptor appendPixelBuffer:pixelBuffer withPresentationTime:hostPTS]) {
    self.appendFailureCount += 1;
    [self setFailureOnSampleQueue:self.writer.error ?:
        captureError(@"AVAssetWriter rejected a video sample")];
    return NO;
  }
  self.lastAppendedHostPTS = hostPTS;
  self.appendedSampleCount += 1;
  double relativePTS = CMTimeGetSeconds(
      CMTimeSubtract(hostPTS, self.sessionStartHostPTS));
  if (!isfinite(relativePTS) || relativePTS < 0.0) {
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:captureError(@"appended video PTS is invalid")];
    return NO;
  }
  [self.appendedRelativePTSSeconds addObject:@(relativePTS)];
  char kindCode = [kind isEqualToString:@"started"] ? 'S' :
                  [kind isEqualToString:@"complete"] ? 'C' :
                  [kind isEqualToString:@"idle"] ? 'I' :
                  [kind isEqualToString:@"recovery"] ? 'R' : 'E';
  [self.appendedPTSTraceRows appendFormat:@"%lld,%c\n",
      (long long)llround(relativePTS * 1000000.0), kindCode];
  if ([kind isEqualToString:@"started"]) self.appendedStartedCount += 1;
  if ([kind isEqualToString:@"complete"]) {
    self.appendedCompleteCount += 1;
    self.lastCompleteDisplayTime =
        CMClockConvertHostTimeToSystemUnits(hostPTS);
    self.lastCompleteCallbackSequence = self.callbackCount;
  }
  if ([kind isEqualToString:@"idle"]) self.appendedIdleCount += 1;
  if ([kind isEqualToString:@"recovery"])
    self.appendedTerminalRecoveryCount += 1;
  if ([kind isEqualToString:@"terminal"]) self.terminalHoldCount += 1;
  return YES;
}

- (BOOL)insertTerminalGapRecoveryAtHostTime:(CMTime)hostPTS
                    visualSignature:(uint64_t)visualSignature
                        contentClass:(CaptureFrameClass)contentClass {
  if (self.terminalGapRecoveries.count != 0 ||
      self.appendedTerminalRecoveryCount != 0) {
    [self setFailureOnSampleQueue:
        captureError(@"capture requires more than one terminal delivery-gap recovery")];
    return NO;
  }
  if (![self appendPixelBuffer:self.lastPixelBuffer
                     atHostTime:hostPTS
                            kind:@"recovery"])
    return NO;
  [self recordAppendedVisualSignature:visualSignature contentClass:contentClass];
  return YES;
}

- (BOOL)commitTerminalGapRecoveryEvidence:(NSDictionary *)recovery {
  if (self.terminalGapRecoveries.count != 0 ||
      self.appendedTerminalRecoveryCount != 1) {
    [self setFailureOnSampleQueue:
        captureError(@"terminal delivery-gap recovery commit ordering is invalid")];
    return NO;
  }
  [self.terminalGapRecoveries addObject:recovery];
  [self.terminalGapRecoveryRows appendFormat:
      @"%@,%@,%@,%@,%@,%@,%@,%@\n",
      recovery[@"recovery_index"], recovery[@"prior_callback_sequence"],
      recovery[@"current_callback_sequence"],
      recovery[@"prior_relative_display_us"],
      recovery[@"current_relative_display_us"],
      recovery[@"inserted_relative_pts_us"],
      recovery[@"prior_content_class_code"],
      recovery[@"prior_visual_signature"]];
  return YES;
}

- (void)recordAppendedVisualSignature:(uint64_t)visualSignature
                         contentClass:(CaptureFrameClass)contentClass {
  if (self.appendedVisualStateInitialized) {
    if (visualSignature != self.lastAppendedVisualSignature)
      self.visualSignatureTransitionCount += 1;
    if (contentClass != self.lastAppendedContentClass)
      self.contentClassTransitionCount += 1;
  }
  self.appendedVisualStateInitialized = YES;
  self.lastAppendedVisualSignature = visualSignature;
  self.lastAppendedContentClass = contentClass;
}

- (void)updateGameplayContentClass:(CaptureFrameClass)frameClass
                        atHostTime:(CMTime)hostPTS {
  self.lastFrameClass = frameClass;
  if (!self.gameplayArmed) return;
  if (frameClass == CaptureFrameClassActive) {
    self.blankRunClass = CaptureFrameClassUnknown;
    self.blankRunStartHostPTS = kCMTimeInvalid;
    return;
  }
  if (frameClass == CaptureFrameClassUnknown) {
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:captureError(@"could not classify captured gameplay pixels")];
    return;
  }
  if (self.blankRunClass != frameClass || !CMTIME_IS_NUMERIC(self.blankRunStartHostPTS)) {
    self.blankRunClass = frameClass;
    self.blankRunStartHostPTS = hostPTS;
  }
  double duration = CMTimeGetSeconds(CMTimeSubtract(hostPTS, self.blankRunStartHostPTS));
  if (!isfinite(duration) || duration < 0.0) {
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:captureError(@"captured blank-run timing is invalid")];
    return;
  }
  if (frameClass == CaptureFrameClassNearWhite)
    self.longestNearWhiteSeconds = MAX(self.longestNearWhiteSeconds, duration);
  if (frameClass == CaptureFrameClassNearBlack)
    self.longestNearBlackSeconds = MAX(self.longestNearBlackSeconds, duration);
  if (frameClass == CaptureFrameClassNeutralBlank)
    self.longestNeutralBlankSeconds = MAX(self.longestNeutralBlankSeconds, duration);
  if (duration > MaximumSustainedBlankSeconds)
    [self setFailureOnSampleQueue:
        captureError([NSString stringWithFormat:
            @"captured gameplay blank class %c persisted %.6f seconds",
            frameClassCode(frameClass), duration])];
}

- (void)finalizeCompactTraceRow:(NSArray *)row
                       hostPTS:(CMTime)hostPTS
                    deliveryLag:(double)deliveryLag
                        service:(double)service {
  double relative = CMTimeGetSeconds(CMTimeSubtract(hostPTS, self.sessionStartHostPTS));
  [self.callbackTraceRows appendFormat:@"%@,%lld,%lld,%lld,%@,%@,%@,%@\n",
      row[0], (long long)llround(relative * 1000000.0),
      (long long)llround(deliveryLag * 1000000.0),
      (long long)llround(service * 1000000.0),
      row[3], row[4], row[5], row[6]];
}

- (void)stream:(SCStream *)stream didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
         ofType:(SCStreamOutputType)type {
  if (type != SCStreamOutputTypeScreen) return;
  CMTime callbackStarted = CMClockGetTime(CMClockGetHostTimeClock());
  self.callbackCount += 1;
  NSMutableArray *traceRow = [@[
    @(self.callbackCount), @(-1),
    @(CMClockConvertHostTimeToSystemUnits(callbackStarted)),
    @"X", @"X", @0, @"0000000000000000",
  ] mutableCopy];
  [self.callbackTrace addObject:traceRow];
  if (!self.acceptingSamples || self.failure) return;
  if (!sampleBuffer || !CMSampleBufferIsValid(sampleBuffer)) {
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:captureError(@"ScreenCaptureKit delivered an invalid sample")];
    return;
  }
  traceRow[5] = @(TraceFlagValidSample);

  NSDictionary *attachment = firstSampleAttachment(sampleBuffer);
  NSNumber *statusNumber = attachment[(id)SCStreamFrameInfoStatus];
  NSNumber *displayTimeNumber = attachment[(id)SCStreamFrameInfoDisplayTime];
  if (![statusNumber isKindOfClass:NSNumber.class] ||
      ![displayTimeNumber isKindOfClass:NSNumber.class]) {
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:
        captureError(@"ScreenCaptureKit sample lacks concrete status or display time")];
    return;
  }

  SCFrameStatus status = (SCFrameStatus)statusNumber.integerValue;
  NSString *statusName = frameStatusName(status);
  traceRow[3] = [NSString stringWithFormat:@"%c", frameStatusCode(status)];
  traceRow[5] = @([traceRow[5] unsignedIntegerValue] | TraceFlagValidStatus);

  CFTypeRef droppedReason = CMGetAttachment(
      sampleBuffer, kCMSampleBufferAttachmentKey_DroppedFrameReason, NULL);
  if (!droppedReason)
    droppedReason = (__bridge CFTypeRef)
        attachment[(__bridge NSString *)kCMSampleBufferAttachmentKey_DroppedFrameReason];
  if (droppedReason) {
    traceRow[5] = @([traceRow[5] unsignedIntegerValue] | TraceFlagDropAttachment);
    NSString *reason = [(__bridge id)droppedReason description] ?: @"unknown";
    self.dropAttachmentCount += 1;
    self.dropReasons[reason] = @([self.dropReasons[reason] unsignedIntegerValue] + 1);
    [self setFailureOnSampleQueue:
        captureError([NSString stringWithFormat:@"ScreenCaptureKit reported a dropped frame: %@",
                                                 reason])];
    return;
  }

  if ([statusName isEqualToString:@"unknown"]) {
    [self incrementStatus:@"unknown"];
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:captureError(@"ScreenCaptureKit reported an unknown status")];
    return;
  }
  [self incrementStatus:statusName];

  uint64_t displayTime = displayTimeNumber.unsignedLongLongValue;
  traceRow[1] = @(displayTime);
  if (displayTime == 0 || (self.lastDisplayTime && displayTime <= self.lastDisplayTime)) {
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:
        captureError(@"ScreenCaptureKit display times must increase strictly")];
    return;
  }
  traceRow[5] = @([traceRow[5] unsignedIntegerValue] | TraceFlagValidDisplayTime |
                  (self.gameplayArmed ? TraceFlagGameplayArmed : 0) |
                  (self.stopRequested ? TraceFlagStopRequested : 0));
  CMTime hostPTS = CMClockMakeHostTimeFromSystemUnits(displayTime);
  double hostPTSSeconds = CMTimeGetSeconds(hostPTS);
  if (!CMTIME_IS_NUMERIC(hostPTS) || !isfinite(hostPTSSeconds)) {
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:captureError(@"ScreenCaptureKit display time is invalid")];
    return;
  }
  NSDictionary *pendingTerminalGapRecovery = nil;
  CMTime pendingTerminalGapRecoveryPTS = kCMTimeInvalid;
  uint64_t pendingTerminalGapPriorVisualSignature = 0;
  CaptureFrameClass pendingTerminalGapPriorContentClass =
      CaptureFrameClassUnknown;
  if (self.lastDisplayTime) {
    CMTime previous = CMClockMakeHostTimeFromSystemUnits(self.lastDisplayTime);
    double gap = CMTimeGetSeconds(CMTimeSubtract(hostPTS, previous));
    self.maximumDisplayGapSeconds = MAX(self.maximumDisplayGapSeconds, gap);
    if (!isfinite(gap) || gap <= 0.0 ||
        (self.startupPhaseComplete && gap > TerminalDeliveryGapRecoveryCeilingSeconds)) {
      [self setFailureOnSampleQueue:
          captureError([NSString stringWithFormat:
              @"ScreenCaptureKit delivery gap %.6f exceeds terminal recovery ceiling %.6f seconds",
              gap, TerminalDeliveryGapRecoveryCeilingSeconds])];
      return;
    }
    double priorRelativeSeconds = CMTimeGetSeconds(
        CMTimeSubtract(previous, self.sessionStartHostPTS));
    double currentRelativeSeconds = CMTimeGetSeconds(
        CMTimeSubtract(hostPTS, self.sessionStartHostPTS));
    long long priorRelativeUS = llround(priorRelativeSeconds * 1000000.0);
    long long currentRelativeUS = llround(currentRelativeSeconds * 1000000.0);
    long long canonicalGapUS = currentRelativeUS - priorRelativeUS;
    if (!isfinite(priorRelativeSeconds) || !isfinite(currentRelativeSeconds) ||
        priorRelativeSeconds < 0.0 || canonicalGapUS <= 0) {
      [self setFailureOnSampleQueue:
          captureError(@"ScreenCaptureKit delivery gap has invalid canonical timing")];
      return;
    }
    if (self.startupPhaseComplete &&
        canonicalGapUS > llround(MaximumDeliveryGapSeconds * 1000000.0)) {
      if (!self.terminalGapRecoveryArmed) {
        [self setFailureOnSampleQueue:
            captureError([NSString stringWithFormat:
                @"ScreenCaptureKit delivery gap %.6f exceeds %.6f seconds outside the proven terminal window",
                gap, MaximumDeliveryGapSeconds])];
        return;
      }
      if (self.terminalGapRecoveries.count != 0 ||
          self.appendedTerminalRecoveryCount != 0) {
        [self setFailureOnSampleQueue:
            captureError(@"capture requires more than one terminal delivery-gap recovery")];
        return;
      }
      if (!self.sessionStarted || !self.lastPixelBuffer ||
          !CMTIME_IS_NUMERIC(self.sessionStartHostPTS) ||
          !CMTIME_IS_NUMERIC(self.lastAppendedHostPTS) ||
          !self.appendedVisualStateInitialized ||
          self.lastFrameClass == CaptureFrameClassUnknown ||
          self.lastVisualSignature == 0) {
        [self setFailureOnSampleQueue:
            captureError(@"ScreenCaptureKit delivery gap lacks a retained prior visual state")];
        return;
      }
      double priorAppendOffset = fabs(CMTimeGetSeconds(
          CMTimeSubtract(self.lastAppendedHostPTS, previous)));
      if (!isfinite(priorAppendOffset) ||
          priorAppendOffset > EncodedPTSNumericalSlackSeconds) {
        [self setFailureOnSampleQueue:
            captureError(@"ScreenCaptureKit delivery gap does not follow its prior callback PTS")];
        return;
      }
      NSUInteger priorCallbackSequence = self.callbackCount - 1;
      long long insertedRelativeUS = priorRelativeUS +
          (currentRelativeUS - priorRelativeUS) / 2;
      pendingTerminalGapRecoveryPTS = CMTimeAdd(
          self.sessionStartHostPTS, CMTimeMake(insertedRelativeUS, 1000000));
      if (insertedRelativeUS <= priorRelativeUS ||
          insertedRelativeUS >= currentRelativeUS ||
          !CMTIME_IS_NUMERIC(pendingTerminalGapRecoveryPTS) ||
          CMTimeCompare(pendingTerminalGapRecoveryPTS, previous) <= 0 ||
          CMTimeCompare(pendingTerminalGapRecoveryPTS, hostPTS) >= 0) {
        [self setFailureOnSampleQueue:
            captureError(@"terminal delivery-gap midpoint is invalid")];
        return;
      }
      NSString *priorClass = [NSString stringWithFormat:@"%c",
          frameClassCode(self.lastFrameClass)];
      NSString *priorSignature = [NSString stringWithFormat:@"%016llx",
          (unsigned long long)self.lastVisualSignature];
      pendingTerminalGapPriorVisualSignature = self.lastVisualSignature;
      pendingTerminalGapPriorContentClass = self.lastFrameClass;
      pendingTerminalGapRecovery = @{
        @"recovery_index": @1,
        @"prior_callback_sequence": @(priorCallbackSequence),
        @"current_callback_sequence": @(self.callbackCount),
        @"prior_relative_display_us": @(priorRelativeUS),
        @"current_relative_display_us": @(currentRelativeUS),
        @"inserted_relative_pts_us": @(insertedRelativeUS),
        @"prior_content_class_code": priorClass,
        @"prior_visual_signature": priorSignature,
      };
    }
  } else {
    self.firstDisplayTime = displayTime;
  }
  self.lastDisplayTime = displayTime;

  double deliveryLag = CMTimeGetSeconds(CMTimeSubtract(callbackStarted, hostPTS));
  if (!isfinite(deliveryLag) || deliveryLag < -0.001) {
    self.invalidSampleCount += 1;
    [self setFailureOnSampleQueue:captureError(@"ScreenCaptureKit callback timing is invalid")];
    return;
  }
  deliveryLag = MAX(0.0, deliveryLag);
  self.maximumCallbackDeliveryLagSeconds =
      MAX(self.maximumCallbackDeliveryLagSeconds, deliveryLag);
  if (deliveryLag > MaximumCallbackDeliveryLagSeconds) {
    [self setFailureOnSampleQueue:
        captureError([NSString stringWithFormat:
            @"ScreenCaptureKit callback lag %.6f exceeds %.6f seconds",
            deliveryLag, MaximumCallbackDeliveryLagSeconds])];
    return;
  }

  if (status == SCFrameStatusBlank || status == SCFrameStatusSuspended) {
    [self setFailureOnSampleQueue:
        captureError([NSString stringWithFormat:@"ScreenCaptureKit reported %@ content",
                                                 statusName])];
    return;
  }
  if (status == SCFrameStatusStopped) {
    if (pendingTerminalGapRecovery) {
      [self setFailureOnSampleQueue:
          captureError(@"terminal delivery-gap recovery ended on a Stopped callback")];
      return;
    }
    traceRow[5] = @([traceRow[5] unsignedIntegerValue] | TraceFlagStopped);
    CMTime callbackFinished = CMClockGetTime(CMClockGetHostTimeClock());
    double service = CMTimeGetSeconds(CMTimeSubtract(callbackFinished, callbackStarted));
    self.maximumCallbackServiceSeconds = MAX(self.maximumCallbackServiceSeconds, service);
    [self finalizeCompactTraceRow:traceRow hostPTS:hostPTS
                       deliveryLag:deliveryLag service:service];
    self.stoppedStatusObserved = YES;
    self.stoppedStatusFollowedRequest = self.stopRequested;
    if (!self.stopRequested)
      [self setFailureOnSampleQueue:captureError(@"ScreenCaptureKit stopped unexpectedly")];
    if (!isfinite(service) || service < 0.0 || service > MaximumCallbackServiceSeconds)
      [self setFailureOnSampleQueue:
          captureError([NSString stringWithFormat:
              @"capture stop callback service %.6f exceeds %.6f seconds",
              service, MaximumCallbackServiceSeconds])];
    return;
  }

  CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
  CaptureFrameClass frameClass = self.lastFrameClass;
  uint64_t visualSignature = self.lastVisualSignature;
  if (status == SCFrameStatusStarted || status == SCFrameStatusComplete) {
    if (!CMSampleBufferDataIsReady(sampleBuffer) || !pixelBuffer ||
        CVPixelBufferGetWidth(pixelBuffer) != self.width ||
        CVPixelBufferGetHeight(pixelBuffer) != self.height ||
        CVPixelBufferGetPixelFormatType(pixelBuffer) !=
            kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
      self.invalidSampleCount += 1;
      [self setFailureOnSampleQueue:
          captureError(@"ScreenCaptureKit delivered an invalid 420v image sample")];
      return;
    }
    traceRow[5] = @([traceRow[5] unsignedIntegerValue] | TraceFlagValidPixel);
    CaptureFrameClassification classification = classifyCentralGameplayPixels(pixelBuffer);
    frameClass = classification.frameClass;
    visualSignature = classification.signature;
    if (self.finalFrozenTailSealed &&
        (frameClass != self.finalFrozenTailContentClass ||
         visualSignature != self.finalFrozenTailVisualSignature)) {
      self.invalidSampleCount += 1;
      [self setFailureOnSampleQueue:
          captureError(@"post-boundary capture callback changed the sealed terminal visual state")];
      return;
    }
    self.lastVisualSignature = visualSignature;
    traceRow[4] = [NSString stringWithFormat:@"%c", frameClassCode(frameClass)];
    traceRow[6] = [NSString stringWithFormat:@"%016llx",
                                            (unsigned long long)visualSignature];
    self.classifiedFrameCount += 1;
    [self updateGameplayContentClass:frameClass atHostTime:hostPTS];
    if (self.failure) return;
    if (status == SCFrameStatusStarted) {
      if (self.sessionStarted) {
        self.invalidSampleCount += 1;
        [self setFailureOnSampleQueue:
            captureError(@"ScreenCaptureKit delivered more than one Started frame")];
        return;
      }
      self.sessionStartHostPTS = hostPTS;
      [self.writer startSessionAtSourceTime:hostPTS];
      self.sessionStarted = YES;
    } else if (!self.sessionStarted) {
      // Some ScreenCaptureKit versions begin a window stream with Complete.
      // Preserve that status in telemetry and use its display time as origin.
      self.sessionStartHostPTS = hostPTS;
      [self.writer startSessionAtSourceTime:hostPTS];
      self.sessionStarted = YES;
    }
    if (pendingTerminalGapRecovery &&
        ![self insertTerminalGapRecoveryAtHostTime:pendingTerminalGapRecoveryPTS
                         visualSignature:pendingTerminalGapPriorVisualSignature
                             contentClass:pendingTerminalGapPriorContentClass])
      return;
    [self replaceLastPixelBuffer:pixelBuffer];
    if (![self appendPixelBuffer:pixelBuffer atHostTime:hostPTS kind:statusName]) return;
    [self recordAppendedVisualSignature:visualSignature contentClass:frameClass];
    traceRow[5] = @([traceRow[5] unsignedIntegerValue] | TraceFlagAppended |
                    (frameClass == CaptureFrameClassActive ? TraceFlagActiveContent : 0));
    if (pendingTerminalGapRecovery &&
        ![self commitTerminalGapRecoveryEvidence:pendingTerminalGapRecovery])
      return;
    [self signalStartupIfNeeded];
  } else if (status == SCFrameStatusIdle) {
    if (!self.sessionStarted || !self.lastPixelBuffer) {
      self.invalidSampleCount += 1;
      [self setFailureOnSampleQueue:
          captureError(@"ScreenCaptureKit delivered Idle before the first image")];
      return;
    }
    if (pixelBuffer && CMSampleBufferDataIsReady(sampleBuffer)) {
      if (CVPixelBufferGetWidth(pixelBuffer) != self.width ||
          CVPixelBufferGetHeight(pixelBuffer) != self.height ||
          CVPixelBufferGetPixelFormatType(pixelBuffer) !=
              kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
        self.invalidSampleCount += 1;
        [self setFailureOnSampleQueue:
            captureError(@"ScreenCaptureKit Idle sample supplied an invalid pixel buffer")];
        return;
      }
      CaptureFrameClassification idleClassification =
          classifyCentralGameplayPixels(pixelBuffer);
      if (idleClassification.frameClass == CaptureFrameClassUnknown ||
          idleClassification.signature != self.lastVisualSignature) {
        self.invalidSampleCount += 1;
        [self setFailureOnSampleQueue:
            captureError(@"ScreenCaptureKit Idle pixels differ from the retained visual state")];
        return;
      }
      traceRow[5] = @([traceRow[5] unsignedIntegerValue] | TraceFlagValidPixel);
    }
    traceRow[4] = [NSString stringWithFormat:@"%c", frameClassCode(frameClass)];
    traceRow[6] = [NSString stringWithFormat:@"%016llx",
                                            (unsigned long long)visualSignature];
    [self updateGameplayContentClass:frameClass atHostTime:hostPTS];
    if (self.failure) return;
    if (pendingTerminalGapRecovery &&
        ![self insertTerminalGapRecoveryAtHostTime:pendingTerminalGapRecoveryPTS
                         visualSignature:pendingTerminalGapPriorVisualSignature
                             contentClass:pendingTerminalGapPriorContentClass])
      return;
    if (![self appendPixelBuffer:self.lastPixelBuffer atHostTime:hostPTS kind:@"idle"]) return;
    [self recordAppendedVisualSignature:visualSignature contentClass:frameClass];
    traceRow[5] = @([traceRow[5] unsignedIntegerValue] | TraceFlagAppended |
                    TraceFlagIdleHold |
                    (frameClass == CaptureFrameClassActive ? TraceFlagActiveContent : 0));
    if (pendingTerminalGapRecovery &&
        ![self commitTerminalGapRecoveryEvidence:pendingTerminalGapRecovery])
      return;
  }
  CMTime callbackFinished = CMClockGetTime(CMClockGetHostTimeClock());
  double service = CMTimeGetSeconds(CMTimeSubtract(callbackFinished, callbackStarted));
  self.maximumCallbackServiceSeconds = MAX(self.maximumCallbackServiceSeconds, service);
  [self finalizeCompactTraceRow:traceRow hostPTS:hostPTS
                     deliveryLag:deliveryLag service:service];
  if (!isfinite(service) || service < 0.0 || service > MaximumCallbackServiceSeconds)
    [self setFailureOnSampleQueue:
        captureError([NSString stringWithFormat:
            @"capture callback service %.6f exceeds %.6f seconds",
            service, MaximumCallbackServiceSeconds])];
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
  dispatch_async(self.sampleQueue, ^{
    self.delegateStopCallbackObserved = YES;
    self.delegateStopErrorObserved = error != nil;
    if (error) [self setFailureOnSampleQueue:error];
    [self signalStartupIfNeeded];
  });
}

- (NSError *)failureSnapshot {
  __block NSError *result = nil;
  dispatch_sync(self.sampleQueue, ^{ result = self.failure; });
  return result;
}

- (double)currentSourceVideoSeconds {
  __block CMTime origin = kCMTimeInvalid;
  __block uint64_t displayTime = 0;
  dispatch_sync(self.sampleQueue, ^{
    origin = self.sessionStartHostPTS;
    displayTime = self.lastDisplayTime;
  });
  if (!CMTIME_IS_NUMERIC(origin) || displayTime == 0) return NAN;
  CMTime latestDisplay = CMClockMakeHostTimeFromSystemUnits(displayTime);
  return MAX(0.0, CMTimeGetSeconds(CMTimeSubtract(latestDisplay, origin)));
}

- (NSDictionary *)boundarySnapshotOnSampleQueue {
    double relativeDisplaySeconds = NAN;
    double displayHostSeconds = NAN;
    double completeDisplayHostSeconds = 0.0;
    double completeRelativeDisplaySeconds = 0.0;
    if (CMTIME_IS_NUMERIC(self.sessionStartHostPTS) && self.lastDisplayTime != 0) {
      CMTime latestDisplay = CMClockMakeHostTimeFromSystemUnits(self.lastDisplayTime);
      relativeDisplaySeconds = MAX(
          0.0, CMTimeGetSeconds(CMTimeSubtract(latestDisplay, self.sessionStartHostPTS)));
      displayHostSeconds = CMTimeGetSeconds(latestDisplay);
    }
    if (self.lastCompleteDisplayTime != 0) {
      CMTime latestCompleteDisplay =
          CMClockMakeHostTimeFromSystemUnits(self.lastCompleteDisplayTime);
      completeDisplayHostSeconds = CMTimeGetSeconds(latestCompleteDisplay);
      if (CMTIME_IS_NUMERIC(self.sessionStartHostPTS))
        completeRelativeDisplaySeconds = MAX(
            0.0, CMTimeGetSeconds(CMTimeSubtract(
                     latestCompleteDisplay, self.sessionStartHostPTS)));
    }
    return @{
      @"callback_count": @(self.callbackCount),
      @"complete_sample_count": @(self.appendedCompleteCount),
      @"idle_sample_count": @(self.appendedIdleCount),
      @"last_display_time_mach": @(self.lastDisplayTime),
      @"last_display_host_seconds": @(displayHostSeconds),
      @"last_complete_display_time_mach": @(self.lastCompleteDisplayTime),
      @"last_complete_display_host_seconds":
          @(completeDisplayHostSeconds),
      @"last_complete_relative_display_seconds":
          @(completeRelativeDisplaySeconds),
      @"last_complete_callback_sequence":
          @(self.lastCompleteCallbackSequence),
      @"last_relative_display_seconds": @(relativeDisplaySeconds),
      @"last_visual_signature":
          [NSString stringWithFormat:@"%016llx",
                                     (unsigned long long)self.lastVisualSignature],
      @"last_content_class":
          [NSString stringWithFormat:@"%c", frameClassCode(self.lastFrameClass)],
      @"visual_signature_transition_count":
          @(self.visualSignatureTransitionCount),
      @"content_class_transition_count":
          @(self.contentClassTransitionCount),
    };
}

- (NSDictionary *)boundarySnapshot {
  __block NSDictionary *result = nil;
  dispatch_sync(self.sampleQueue, ^{
    result = [self boundarySnapshotOnSampleQueue];
  });
  return result;
}

- (NSDictionary *)boundarySnapshotAndArmTerminalGapRecovery {
  __block NSDictionary *result = nil;
  dispatch_sync(self.sampleQueue, ^{
    if (self.terminalGapRecoveryArmed) {
      [self setFailureOnSampleQueue:
          captureError(@"terminal delivery-gap recovery was armed more than once")];
      return;
    }
    result = [self boundarySnapshotOnSampleQueue];
    self.terminalGapRecoveryArmed = YES;
  });
  return result;
}

- (NSDictionary *)validateFinalFrozenTailFromAccepted:(NSDictionary *)accepted
                                        throughBoundary:(NSDictionary *)finalBoundary {
  __block NSDictionary *result = nil;
  dispatch_sync(self.sampleQueue, ^{
    NSUInteger firstSequence =
        [accepted[@"last_complete_callback_sequence"] unsignedIntegerValue];
    NSUInteger lastSequence =
        [finalBoundary[@"callback_count"] unsignedIntegerValue];
    NSString *acceptedSignature = accepted[@"last_visual_signature"];
    NSString *acceptedClass = accepted[@"last_content_class"];
    NSUInteger completeCount = 0;
    NSUInteger idleCount = 0;
    NSUInteger forbiddenCount = 0;
    NSUInteger visualTransitions = 0;
    NSUInteger classTransitions = 0;
    long long firstRelativeUS = -1;
    long long lastRelativeUS = -1;
    NSString *previousSignature = nil;
    NSString *previousClass = nil;
    BOOL passed = firstSequence >= 1 && lastSequence > firstSequence &&
        lastSequence <= self.callbackTrace.count &&
        acceptedSignature.length == 16 &&
        ![acceptedSignature isEqualToString:@"0000000000000000"] &&
        [@[@"A", @"W", @"K", @"N"] containsObject:acceptedClass];
    NSUInteger requiredFlags = TraceFlagValidSample | TraceFlagValidStatus |
        TraceFlagValidDisplayTime | TraceFlagAppended | TraceFlagGameplayArmed;
    NSUInteger forbiddenFlags = TraceFlagDropAttachment | TraceFlagStopped |
        TraceFlagStopRequested;
    if (passed) {
      for (NSUInteger sequence = firstSequence; sequence <= lastSequence;
           ++sequence) {
        NSArray *row = self.callbackTrace[sequence - 1];
        NSString *status = row[3];
        NSString *contentClass = row[4];
        NSUInteger flags = [row[5] unsignedIntegerValue];
        NSString *signature = row[6];
        uint64_t displayTime = [row[1] unsignedLongLongValue];
        long long relativeUS = llround(CMTimeGetSeconds(CMTimeSubtract(
            CMClockMakeHostTimeFromSystemUnits(displayTime),
            self.sessionStartHostPTS)) * 1000000.0);
        if (sequence == firstSequence)
          firstRelativeUS = relativeUS;
        if (lastRelativeUS >= 0 && relativeUS <= lastRelativeUS)
          passed = NO;
        lastRelativeUS = relativeUS;
        if (previousSignature && ![signature isEqualToString:previousSignature])
          visualTransitions += 1;
        if (previousClass && ![contentClass isEqualToString:previousClass])
          classTransitions += 1;
        previousSignature = signature;
        previousClass = contentClass;
        BOOL retained = [@[@"C", @"I"] containsObject:status] &&
            [contentClass isEqualToString:acceptedClass] &&
            [signature isEqualToString:acceptedSignature] &&
            (flags & requiredFlags) == requiredFlags &&
            !(flags & forbiddenFlags) &&
            (([status isEqualToString:@"C"] &&
              (flags & TraceFlagValidPixel) && !(flags & TraceFlagIdleHold)) ||
             ([status isEqualToString:@"I"] &&
              (flags & TraceFlagIdleHold))) &&
            (!!(flags & TraceFlagActiveContent) ==
             [acceptedClass isEqualToString:@"A"]);
        if (!retained) {
          forbiddenCount += 1;
          passed = NO;
        }
        if ([status isEqualToString:@"C"])
          completeCount += 1;
        else if ([status isEqualToString:@"I"])
          idleCount += 1;
      }
    }
    if (firstSequence <= self.callbackTrace.count &&
        ![self.callbackTrace[firstSequence - 1][3] isEqualToString:@"C"])
      passed = NO;
    if (![finalBoundary[@"last_visual_signature"]
            isEqualToString:acceptedSignature] ||
        ![finalBoundary[@"last_content_class"]
            isEqualToString:acceptedClass] ||
        [finalBoundary[@"visual_signature_transition_count"]
            unsignedIntegerValue] !=
            [accepted[@"visual_signature_transition_count"]
                unsignedIntegerValue] ||
        [finalBoundary[@"content_class_transition_count"]
            unsignedIntegerValue] !=
            [accepted[@"content_class_transition_count"]
                unsignedIntegerValue])
      passed = NO;
    NSUInteger rowCount = lastSequence >= firstSequence ?
        lastSequence - firstSequence + 1 : 0;
    if (rowCount != completeCount + idleCount || visualTransitions != 0 ||
        classTransitions != 0)
      passed = NO;
    result = @{
      @"method": @"retained-complete-idle-final-frozen-tail-interval-v1",
      @"first_callback_sequence": @(firstSequence),
      @"last_callback_sequence": @(lastSequence),
      @"row_count": @(rowCount),
      @"post_accept_callback_count": @(rowCount > 0 ? rowCount - 1 : 0),
      @"complete_callback_count": @(completeCount),
      @"idle_callback_count": @(idleCount),
      @"appended_callback_count": @(completeCount + idleCount),
      @"forbidden_callback_count": @(forbiddenCount),
      @"first_relative_display_us": @(firstRelativeUS),
      @"last_relative_display_us": @(lastRelativeUS),
      @"display_interval_us": @(firstRelativeUS >= 0 && lastRelativeUS >= firstRelativeUS ?
          lastRelativeUS - firstRelativeUS : -1),
      @"accepted_visual_signature": acceptedSignature ?: @"",
      @"accepted_content_class": acceptedClass ?: @"",
      @"visual_signature_transition_delta": @(visualTransitions),
      @"content_class_transition_delta": @(classTransitions),
      @"passed": jsonBoolean(passed),
    };
    if (!passed)
      [self setFailureOnSampleQueue:
          captureError(@"terminal final frozen-tail callback trace is not retained")];
  });
  return result;
}

- (NSDictionary *)validateSealedStopSuffixFromFinalBoundary:(NSDictionary *)finalBoundary
                                             acceptedComplete:(NSDictionary *)accepted {
  __block NSDictionary *result = nil;
  dispatch_sync(self.sampleQueue, ^{
    NSUInteger boundarySequence =
        [finalBoundary[@"callback_count"] unsignedIntegerValue];
    NSUInteger firstSequence = boundarySequence + 1;
    NSUInteger lastSequence = self.callbackCount;
    NSString *acceptedSignature = accepted[@"last_visual_signature"];
    NSString *acceptedClass = accepted[@"last_content_class"];
    NSUInteger completeCount = 0;
    NSUInteger idleCount = 0;
    NSUInteger stoppedCount = 0;
    NSUInteger forbiddenCount = 0;
    NSUInteger lastImageSequence = boundarySequence;
    long long firstRelativeUS = -1;
    long long lastRelativeUS = -1;
    BOOL passed = boundarySequence >= 1 &&
        boundarySequence <= lastSequence && self.finalFrozenTailSealed &&
        self.finalFrozenTailVisualSignature != 0 &&
        self.finalFrozenTailContentClass != CaptureFrameClassUnknown &&
        [acceptedSignature isEqualToString:[NSString stringWithFormat:@"%016llx",
            (unsigned long long)self.finalFrozenTailVisualSignature]] &&
        [acceptedClass isEqualToString:[NSString stringWithFormat:@"%c",
            frameClassCode(self.finalFrozenTailContentClass)]];
    NSUInteger commonFlags = TraceFlagValidSample | TraceFlagValidStatus |
        TraceFlagValidDisplayTime;
    for (NSUInteger sequence = firstSequence;
         passed && sequence <= lastSequence; ++sequence) {
      NSArray *row = self.callbackTrace[sequence - 1];
      NSString *status = row[3];
      NSString *contentClass = row[4];
      NSUInteger flags = [row[5] unsignedIntegerValue];
      NSString *signature = row[6];
      uint64_t displayTime = [row[1] unsignedLongLongValue];
      long long relativeUS = llround(CMTimeGetSeconds(CMTimeSubtract(
          CMClockMakeHostTimeFromSystemUnits(displayTime),
          self.sessionStartHostPTS)) * 1000000.0);
      if (sequence == firstSequence) firstRelativeUS = relativeUS;
      if (lastRelativeUS >= 0 && relativeUS <= lastRelativeUS) passed = NO;
      lastRelativeUS = relativeUS;
      BOOL commonValid = (flags & commonFlags) == commonFlags &&
          !(flags & TraceFlagDropAttachment) &&
          !(flags & TraceFlagGameplayArmed);
      if ([status isEqualToString:@"C"] || [status isEqualToString:@"I"]) {
        BOOL imageValid = commonValid && (flags & TraceFlagAppended) &&
            !(flags & TraceFlagStopped) &&
            [contentClass isEqualToString:acceptedClass] &&
            [signature isEqualToString:acceptedSignature] &&
            (!!(flags & TraceFlagActiveContent) ==
             [acceptedClass isEqualToString:@"A"]);
        if ([status isEqualToString:@"C"])
          imageValid = imageValid && (flags & TraceFlagValidPixel) &&
              !(flags & TraceFlagIdleHold);
        else
          imageValid = imageValid && (flags & TraceFlagIdleHold);
        if (!imageValid) {
          forbiddenCount += 1;
          passed = NO;
        }
        if ([status isEqualToString:@"C"])
          completeCount += 1;
        else
          idleCount += 1;
        lastImageSequence = sequence;
      } else if ([status isEqualToString:@"T"]) {
        BOOL stoppedValid = commonValid && sequence == lastSequence &&
            (flags & TraceFlagStopped) && (flags & TraceFlagStopRequested) &&
            !(flags & (TraceFlagAppended | TraceFlagIdleHold |
                       TraceFlagValidPixel | TraceFlagActiveContent)) &&
            [contentClass isEqualToString:@"X"] &&
            [signature isEqualToString:@"0000000000000000"];
        if (!stoppedValid) {
          forbiddenCount += 1;
          passed = NO;
        }
        stoppedCount += 1;
      } else {
        forbiddenCount += 1;
        passed = NO;
      }
    }
    NSUInteger rowCount = lastSequence - boundarySequence;
    if (rowCount != completeCount + idleCount + stoppedCount ||
        stoppedCount > 1)
      passed = NO;
    result = @{
      @"method": @"retained-complete-idle-optional-stopped-sealed-suffix-v1",
      @"boundary_callback_sequence": @(boundarySequence),
      @"first_callback_sequence": @(rowCount ? firstSequence : 0),
      @"last_callback_sequence": @(lastSequence),
      @"row_count": @(rowCount),
      @"complete_callback_count": @(completeCount),
      @"idle_callback_count": @(idleCount),
      @"stopped_callback_count": @(stoppedCount),
      @"appended_callback_count": @(completeCount + idleCount),
      @"forbidden_callback_count": @(forbiddenCount),
      @"last_image_callback_sequence": @(lastImageSequence),
      @"first_relative_display_us": @(firstRelativeUS),
      @"last_relative_display_us": @(lastRelativeUS),
      @"accepted_visual_signature": acceptedSignature ?: @"",
      @"accepted_content_class": acceptedClass ?: @"",
      @"passed": jsonBoolean(passed),
    };
    if (!passed)
      [self setFailureOnSampleQueue:
          captureError(@"post-boundary sealed stop suffix is invalid")];
  });
  return result;
}

- (NSError *)validateTerminalGapRecoveriesFromBarrier:(NSDictionary *)barrier
                                      acceptedComplete:(NSDictionary *)accepted
                                   throughFinalFrozenTail:(NSDictionary *)finalFrozenTail {
  __block NSError *result = nil;
  dispatch_sync(self.sampleQueue, ^{
    NSUInteger barrierSequence = [barrier[@"callback_count"] unsignedIntegerValue];
    NSUInteger acceptedSequence =
        [accepted[@"last_complete_callback_sequence"] unsignedIntegerValue];
    NSUInteger finalFrozenTailSequence =
        [finalFrozenTail[@"callback_count"] unsignedIntegerValue];
    long long barrierRelativeUS = llround(
        [barrier[@"last_relative_display_seconds"] doubleValue] * 1000000.0);
    long long finalFrozenTailRelativeUS = llround(
        [finalFrozenTail[@"last_relative_display_seconds"] doubleValue] * 1000000.0);
    BOOL withinTerminalWindow = barrierSequence >= 1 &&
        acceptedSequence > barrierSequence &&
        finalFrozenTailSequence >= acceptedSequence &&
        finalFrozenTailSequence <= self.callbackTrace.count &&
        !self.terminalGapRecoveryArmed;
    BOOL visualConsistent = YES;
    BOOL encodedGapsSplit = YES;
    NSUInteger controlledTransitionCount = 0;
    NSUInteger retainedStateCount = 0;
    NSUInteger rawOverLimitGapCount = 0;
    NSUInteger rawOverLimitPriorSequence = 0;
    NSUInteger rawOverLimitCurrentSequence = 0;
    for (NSUInteger index = 1; index < self.callbackTrace.count; ++index) {
      NSArray *priorRow = self.callbackTrace[index - 1];
      NSArray *currentRow = self.callbackTrace[index];
      uint64_t priorDisplayTime = [priorRow[1] unsignedLongLongValue];
      uint64_t currentDisplayTime = [currentRow[1] unsignedLongLongValue];
      if (priorDisplayTime == 0 || currentDisplayTime <= priorDisplayTime) {
        encodedGapsSplit = NO;
        continue;
      }
      double rawGap = CMTimeGetSeconds(CMTimeSubtract(
          CMClockMakeHostTimeFromSystemUnits(currentDisplayTime),
          CMClockMakeHostTimeFromSystemUnits(priorDisplayTime)));
      long long priorRelativeUS = llround(CMTimeGetSeconds(CMTimeSubtract(
          CMClockMakeHostTimeFromSystemUnits(priorDisplayTime),
          self.sessionStartHostPTS)) * 1000000.0);
      long long currentRelativeUS = llround(CMTimeGetSeconds(CMTimeSubtract(
          CMClockMakeHostTimeFromSystemUnits(currentDisplayTime),
          self.sessionStartHostPTS)) * 1000000.0);
      long long canonicalGapUS = currentRelativeUS - priorRelativeUS;
      if (!isfinite(rawGap) || rawGap <= 0.0) {
        encodedGapsSplit = NO;
        continue;
      }
      if (strictCapturePair(index + 1, self.startupCallbackCount) &&
          canonicalGapUS > llround(MaximumDeliveryGapSeconds * 1000000.0)) {
        rawOverLimitGapCount += 1;
        rawOverLimitPriorSequence = index;
        rawOverLimitCurrentSequence = index + 1;
        if (rawGap > TerminalDeliveryGapRecoveryCeilingSeconds)
          encodedGapsSplit = NO;
      }
    }
    NSUInteger expectedRecoveryIndex = 1;
    for (NSDictionary *recovery in self.terminalGapRecoveries) {
      NSUInteger recoveryIndex =
          [recovery[@"recovery_index"] unsignedIntegerValue];
      NSUInteger priorSequence =
          [recovery[@"prior_callback_sequence"] unsignedIntegerValue];
      NSUInteger currentSequence =
          [recovery[@"current_callback_sequence"] unsignedIntegerValue];
      long long priorRelativeUS =
          [recovery[@"prior_relative_display_us"] longLongValue];
      long long currentRelativeUS =
          [recovery[@"current_relative_display_us"] longLongValue];
      long long insertedRelativeUS =
          [recovery[@"inserted_relative_pts_us"] longLongValue];
      if (recoveryIndex != expectedRecoveryIndex || priorSequence < 1 ||
          currentSequence != priorSequence + 1 ||
          currentSequence > self.callbackTrace.count) {
        withinTerminalWindow = NO;
        visualConsistent = NO;
        encodedGapsSplit = NO;
        expectedRecoveryIndex += 1;
        continue;
      }
      expectedRecoveryIndex += 1;
      NSArray *priorRow = self.callbackTrace[priorSequence - 1];
      NSArray *currentRow = self.callbackTrace[currentSequence - 1];
      uint64_t priorDisplayTime = [priorRow[1] unsignedLongLongValue];
      uint64_t currentDisplayTime = [currentRow[1] unsignedLongLongValue];
      CMTime priorDisplayPTS =
          CMClockMakeHostTimeFromSystemUnits(priorDisplayTime);
      CMTime currentDisplayPTS =
          CMClockMakeHostTimeFromSystemUnits(currentDisplayTime);
      long long tracedPriorRelativeUS = llround(CMTimeGetSeconds(CMTimeSubtract(
          priorDisplayPTS, self.sessionStartHostPTS)) * 1000000.0);
      long long tracedCurrentRelativeUS = llround(CMTimeGetSeconds(CMTimeSubtract(
          currentDisplayPTS, self.sessionStartHostPTS)) * 1000000.0);
      double rawGap = CMTimeGetSeconds(
          CMTimeSubtract(currentDisplayPTS, priorDisplayPTS));
      BOOL endpointWithinWindow =
          priorSequence >= barrierSequence &&
          currentSequence <= finalFrozenTailSequence &&
          priorRelativeUS >= barrierRelativeUS &&
          currentRelativeUS <= finalFrozenTailRelativeUS;
      if (!endpointWithinWindow) withinTerminalWindow = NO;
      if (priorRelativeUS != tracedPriorRelativeUS ||
          currentRelativeUS != tracedCurrentRelativeUS ||
          !isfinite(rawGap) ||
          currentRelativeUS - priorRelativeUS <=
              llround(MaximumDeliveryGapSeconds * 1000000.0) ||
          rawGap > TerminalDeliveryGapRecoveryCeilingSeconds ||
          insertedRelativeUS <= priorRelativeUS ||
          insertedRelativeUS >= currentRelativeUS ||
          llabs(2 * insertedRelativeUS - priorRelativeUS -
                currentRelativeUS) > 1 ||
          insertedRelativeUS - priorRelativeUS >
              llround(MaximumDeliveryGapSeconds * 1000000.0) ||
          currentRelativeUS - insertedRelativeUS >
              llround(MaximumDeliveryGapSeconds * 1000000.0))
        encodedGapsSplit = NO;
      NSString *priorStatus = priorRow[3];
      NSString *currentStatus = currentRow[3];
      NSString *priorClass = priorRow[4];
      NSString *currentClass = currentRow[4];
      NSUInteger priorFlags = [priorRow[5] unsignedIntegerValue];
      NSUInteger currentFlags = [currentRow[5] unsignedIntegerValue];
      NSString *priorSignature = priorRow[6];
      NSString *currentSignature = currentRow[6];
      NSUInteger requiredFlags = TraceFlagValidSample | TraceFlagValidStatus |
          TraceFlagValidDisplayTime | TraceFlagAppended | TraceFlagGameplayArmed;
      NSUInteger forbiddenFlags = TraceFlagDropAttachment | TraceFlagStopped |
          TraceFlagStopRequested;
      BOOL endpointFlagsValid =
          [@[@"C", @"I"] containsObject:priorStatus] &&
          [@[@"C", @"I"] containsObject:currentStatus] &&
          (priorFlags & requiredFlags) == requiredFlags &&
          (currentFlags & requiredFlags) == requiredFlags &&
          !(priorFlags & forbiddenFlags) && !(currentFlags & forbiddenFlags);
      BOOL priorIdentityValid =
          [priorClass isEqualToString:recovery[@"prior_content_class_code"]] &&
          [priorSignature isEqualToString:recovery[@"prior_visual_signature"]] &&
          [@[@"A", @"W", @"K", @"N"] containsObject:priorClass] &&
          priorSignature.length == 16 &&
          ![priorSignature isEqualToString:@"0000000000000000"];
      BOOL retained = [priorClass isEqualToString:currentClass] &&
          [priorSignature isEqualToString:currentSignature];
      BOOL controlledTransition = !retained &&
          currentSequence == acceptedSequence &&
          [currentStatus isEqualToString:@"C"];
      if (retained)
        retainedStateCount += 1;
      else if (controlledTransition)
        controlledTransitionCount += 1;
      if (!endpointFlagsValid || !priorIdentityValid ||
          (!retained && !controlledTransition))
        visualConsistent = NO;
    }
    if (controlledTransitionCount > 1) visualConsistent = NO;
    BOOL countsMatched = self.terminalGapRecoveries.count <= 1 &&
        self.appendedTerminalRecoveryCount == self.terminalGapRecoveries.count &&
        rawOverLimitGapCount == self.terminalGapRecoveries.count;
    if (rawOverLimitGapCount == 1 && self.terminalGapRecoveries.count == 1) {
      NSDictionary *recovery = self.terminalGapRecoveries.firstObject;
      countsMatched = countsMatched &&
          [recovery[@"prior_callback_sequence"] unsignedIntegerValue] ==
              rawOverLimitPriorSequence &&
          [recovery[@"current_callback_sequence"] unsignedIntegerValue] ==
              rawOverLimitCurrentSequence;
    }
    BOOL passed = withinTerminalWindow && visualConsistent &&
        encodedGapsSplit && countsMatched;
    self.terminalGapRecoveryMetadata = @{
      @"method": @"retained-prior-pixel-terminal-window-v1",
      @"encoding": @"recovery-index,prior-callback-sequence,current-callback-sequence,prior-relative-display-us,current-relative-display-us,inserted-relative-pts-us,prior-content-class-code,prior-visual-fnv64-v1",
      @"maximum_allowed_normal_gap_seconds": @(MaximumDeliveryGapSeconds),
      @"maximum_allowed_recoverable_gap_seconds":
          @(TerminalDeliveryGapRecoveryCeilingSeconds),
      @"recovery_interval_count": @(self.terminalGapRecoveries.count),
      @"raw_over_limit_gap_count": @(rawOverLimitGapCount),
      @"inserted_sample_count": @(self.appendedTerminalRecoveryCount),
      @"row_count": @(self.terminalGapRecoveries.count),
      @"rows": [self.terminalGapRecoveryRows copy],
      @"terminal_barrier_callback_sequence": @(barrierSequence),
      @"accepted_terminal_complete_callback_sequence": @(acceptedSequence),
      @"final_frozen_tail_callback_sequence": @(finalFrozenTailSequence),
      @"terminal_barrier_relative_display_us": @(barrierRelativeUS),
      @"final_frozen_tail_relative_display_us": @(finalFrozenTailRelativeUS),
      @"controlled_transition_recovery_count": @(controlledTransitionCount),
      @"retained_state_recovery_count": @(retainedStateCount),
      @"all_recoveries_within_terminal_window": jsonBoolean(withinTerminalWindow),
      @"all_recoveries_visual_consistent": jsonBoolean(visualConsistent),
      @"encoded_gaps_split": jsonBoolean(encodedGapsSplit),
      @"validated_after_stream_stop": @YES,
      @"passed": jsonBoolean(passed),
    };
    if (!passed) {
      result = captureError([NSString stringWithFormat:
          @"terminal delivery-gap recovery proof failed: intervals=%zu inserted=%zu within=%d visual=%d encoded=%d controlled=%zu retained=%zu",
          self.terminalGapRecoveries.count,
          self.appendedTerminalRecoveryCount,
          withinTerminalWindow, visualConsistent, encodedGapsSplit,
          controlledTransitionCount, retainedStateCount]);
      [self setFailureOnSampleQueue:result];
    }
  });
  return result;
}

- (BOOL)waitForCallbackAfterSequence:(NSUInteger)sequence
                             timeout:(double)timeoutSeconds {
  NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeoutSeconds];
  while ([deadline timeIntervalSinceNow] > 0.0) {
    __block BOOL advanced = NO;
    __block BOOL failed = NO;
    dispatch_sync(self.sampleQueue, ^{
      advanced = self.callbackCount > sequence;
      failed = self.failure != nil;
    });
    if (advanced) return YES;
    if (failed) return NO;
    [NSThread sleepForTimeInterval:0.002];
  }
  return NO;
}

- (NSDictionary *)armGameplayAtHostTime:(CMTime)hostPTS {
  __block NSDictionary *boundary = nil;
  dispatch_sync(self.sampleQueue, ^{
    if (self.startupPhaseComplete || self.failure || !self.lastPixelBuffer ||
        self.lastVisualSignature == 0 || self.lastFrameClass == CaptureFrameClassUnknown) {
      [self setFailureOnSampleQueue:captureError(@"invalid or repeated startup phase boundary")];
      return;
    }
    boundary = [self boundarySnapshotOnSampleQueue];
    self.startupCallbackCount = self.callbackCount;
    self.startupSourceVideoSeconds = [boundary[@"last_relative_display_seconds"] doubleValue];
    self.startupPhaseComplete = YES;
    self.gameplayArmed = YES;
    self.blankRunClass = CaptureFrameClassUnknown;
    self.blankRunStartHostPTS = kCMTimeInvalid;
    if (self.lastFrameClass != CaptureFrameClassActive &&
        self.lastFrameClass != CaptureFrameClassUnknown) {
      self.blankRunClass = self.lastFrameClass;
      self.blankRunStartHostPTS = hostPTS;
    }
  });
  return boundary;
}

- (NSDictionary *)disarmGameplayReturningBoundary {
  __block NSDictionary *result = nil;
  dispatch_sync(self.sampleQueue, ^{
    CMTime hostPTS = CMClockGetTime(CMClockGetHostTimeClock());
    if (self.lastDisplayTime != 0)
      hostPTS = CMTimeMaximum(
          hostPTS, CMClockMakeHostTimeFromSystemUnits(self.lastDisplayTime));
    if (self.gameplayArmed && self.blankRunClass != CaptureFrameClassUnknown)
      [self updateGameplayContentClass:self.blankRunClass atHostTime:hostPTS];
    result = [self boundarySnapshotOnSampleQueue];
    self.finalFrozenTailSealed = YES;
    self.finalFrozenTailVisualSignature = self.lastVisualSignature;
    self.finalFrozenTailContentClass = self.lastFrameClass;
    self.terminalGapRecoveryArmed = NO;
    self.gameplayArmed = NO;
  });
  return result;
}

- (void)prepareToStopAtHostTime:(CMTime)stopHostPTS {
  dispatch_sync(self.sampleQueue, ^{
    self.stopRequested = YES;
    self.stopHostPTS = stopHostPTS;
  });
}

- (void)recordStopCompletionWithError:(NSError *)error {
  dispatch_sync(self.sampleQueue, ^{
    if (self.stopCompletionObserved) {
      [self setFailureOnSampleQueue:
          captureError(@"ScreenCaptureKit stop completion ran more than once")];
      return;
    }
    self.stopCompletionObserved = YES;
    self.stopCompletionSucceeded = error == nil;
    self.stopCompletionFollowedRequest = self.stopRequested;
    if (!self.stopCompletionFollowedRequest)
      [self setFailureOnSampleQueue:
          captureError(@"ScreenCaptureKit stop completed before the explicit stop request")];
    if (error) [self setFailureOnSampleQueue:error];
  });
}

- (void)finishWritingAtHostTime:(CMTime)stopHostPTS {
  dispatch_sync(self.sampleQueue, ^{
    if (self.failure) return;
    self.acceptingSamples = NO;
    if (!self.sessionStarted || !self.lastPixelBuffer ||
        !CMTIME_IS_NUMERIC(self.lastAppendedHostPTS)) {
      [self setFailureOnSampleQueue:captureError(@"capture ended without a complete video sample")];
      return;
    }
    CMTime lastEventPTS = CMClockMakeHostTimeFromSystemUnits(self.lastDisplayTime);
    self.terminalCoverageGapSeconds =
        MAX(0.0, CMTimeGetSeconds(CMTimeSubtract(stopHostPTS, lastEventPTS)));
    if (!isfinite(self.terminalCoverageGapSeconds) ||
        self.terminalCoverageGapSeconds > MaximumDeliveryGapSeconds) {
      [self setFailureOnSampleQueue:
          captureError([NSString stringWithFormat:
              @"terminal capture coverage gap %.6f exceeds %.6f seconds",
              self.terminalCoverageGapSeconds, MaximumDeliveryGapSeconds])];
      return;
    }
    CMTime oneFrame = CMTimeMake(1, self.requestedCaptureFPS);
    CMTime terminalPTS = CMTimeMaximum(stopHostPTS,
                                      CMTimeAdd(self.lastAppendedHostPTS, oneFrame));
    if (![self appendPixelBuffer:self.lastPixelBuffer atHostTime:terminalPTS kind:@"terminal"])
      return;
    [self.writer endSessionAtSourceTime:CMTimeAdd(terminalPTS, oneFrame)];
    [self.videoInput markAsFinished];
    [self.writer finishWritingWithCompletionHandler:^{
      dispatch_semaphore_signal(self.writerFinished);
    }];
  });
  if ([self failureSnapshot]) return;
  dispatch_semaphore_wait(self.writerFinished, DISPATCH_TIME_FOREVER);
  if (self.writer.status != AVAssetWriterStatusCompleted) {
    dispatch_sync(self.sampleQueue, ^{
      [self setFailureOnSampleQueue:self.writer.error ?:
          captureError(@"AVAssetWriter did not finish the isolated-window video")];
    });
  }
}

- (NSArray<NSNumber *> *)appendedPTSSequenceSnapshot {
  __block NSArray<NSNumber *> *result = nil;
  dispatch_sync(self.sampleQueue, ^{ result = [self.appendedRelativePTSSeconds copy]; });
  return result;
}

- (NSDictionary *)deliveryMetadataWithPostWriteAudit:(NSDictionary *)postWriteAudit {
  __block NSDictionary *result = nil;
  dispatch_sync(self.sampleQueue, ^{
    double firstPTS = 0.0;
    double lastPTS = CMTimeGetSeconds(
        CMTimeSubtract(self.lastAppendedHostPTS, self.sessionStartHostPTS));
    BOOL accountingPassed = self.appendedSampleCount ==
        self.appendedStartedCount + self.appendedCompleteCount +
        self.appendedIdleCount + self.appendedTerminalRecoveryCount +
        self.terminalHoldCount;
    result = @{
      @"schema": @"sck-stream-output-avassetwriter-v4",
      @"startup_phase": @{
        @"method": @"validated-opening-preroll-v1",
        @"last_callback_sequence": @(self.startupCallbackCount),
        @"source_video_seconds": @(self.startupSourceVideoSeconds),
        @"cutoff_immutable": @YES,
        @"complete": @(self.startupPhaseComplete),
      },
      @"status_source": @"SCStreamFrameInfoStatus",
      @"time_source": @"SCStreamFrameInfoDisplayTime-mach-absolute",
      @"requested_capture_fps": @(self.requestedCaptureFPS),
      @"queue_depth": @8,
      @"pixel_format": @"420v",
      @"writer_codec": @"avc1",
      @"writer_expected_source_fps": @(self.requestedCaptureFPS),
      @"status_counts": [self.statusCounts copy],
      @"callback_count": @(self.callbackCount),
      @"drop_attachment_count": @(self.dropAttachmentCount),
      @"drop_reasons": [self.dropReasons copy],
      @"writer_backpressure_count": @(self.writerBackpressureCount),
      @"append_failure_count": @(self.appendFailureCount),
      @"invalid_sample_count": @(self.invalidSampleCount),
      @"appended_started_samples": @(self.appendedStartedCount),
      @"appended_complete_samples": @(self.appendedCompleteCount),
      @"appended_idle_hold_samples": @(self.appendedIdleCount),
      @"appended_terminal_recovery_samples":
          @(self.appendedTerminalRecoveryCount),
      @"terminal_hold_samples": @(self.terminalHoldCount),
      @"appended_sample_count": @(self.appendedSampleCount),
      @"maximum_display_time_gap_seconds": @(self.maximumDisplayGapSeconds),
      @"maximum_callback_delivery_lag_seconds":
          @(self.maximumCallbackDeliveryLagSeconds),
      @"maximum_callback_service_seconds": @(self.maximumCallbackServiceSeconds),
      @"terminal_coverage_gap_seconds": @(self.terminalCoverageGapSeconds),
      @"terminal_gap_recovery": self.terminalGapRecoveryMetadata,
      @"content_classification": @{
        @"method": @"central-420v-luma-grid-v1",
        @"classified_frame_count": @(self.classifiedFrameCount),
        @"near_white_luma_minimum": @(NearWhiteVideoLumaMinimum),
        @"near_black_luma_maximum": @(NearBlackVideoLumaMaximum),
        @"required_blank_pixel_fraction": @(RequiredBlankPixelFraction),
        @"maximum_allowed_sustained_blank_seconds": @(MaximumSustainedBlankSeconds),
        @"longest_near_white_seconds": @(self.longestNearWhiteSeconds),
        @"longest_near_black_seconds": @(self.longestNearBlackSeconds),
        @"longest_neutral_blank_seconds": @(self.longestNeutralBlankSeconds),
        @"passed": jsonBoolean(!self.failure),
      },
      @"callback_trace": @{
        @"encoding": @"sequence,relative-display-us,delivery-lag-us,service-us,status,class,flags,visual-fnv64-v1",
        @"fields": @[@"sequence", @"relative_display_us", @"delivery_lag_us",
                       @"service_us", @"status_code", @"content_class_code",
                       @"outcome_flags", @"visual_signature"],
        @"status_codes": @{
          @"S": @"started", @"C": @"complete", @"I": @"idle",
          @"B": @"blank", @"U": @"suspended", @"T": @"stopped", @"X": @"unknown",
        },
        @"content_class_codes": @{
          @"A": @"active", @"W": @"near-white", @"K": @"near-black",
          @"N": @"neutral-blank", @"X": @"unknown",
        },
        @"flag_bits": @{
          @"valid_sample": @(TraceFlagValidSample),
          @"valid_status": @(TraceFlagValidStatus),
          @"valid_display_time": @(TraceFlagValidDisplayTime),
          @"drop_attachment": @(TraceFlagDropAttachment),
          @"valid_pixel": @(TraceFlagValidPixel),
          @"appended": @(TraceFlagAppended),
          @"idle_hold": @(TraceFlagIdleHold),
          @"stopped": @(TraceFlagStopped),
          @"gameplay_armed": @(TraceFlagGameplayArmed),
          @"active_content": @(TraceFlagActiveContent),
          @"stop_requested": @(TraceFlagStopRequested),
        },
        @"row_count": @(self.callbackCount),
        @"rows": [self.callbackTraceRows copy],
      },
      @"appended_pts_trace": @{
        @"encoding": @"relative-pts-us,kind-code-v2",
        @"row_count": @(self.appendedSampleCount),
        @"rows": [self.appendedPTSTraceRows copy],
      },
      @"stop_ordering": @{
        @"stop_requested": @(self.stopRequested),
        @"stop_completion_observed": @(self.stopCompletionObserved),
        @"stop_completion_succeeded": @(self.stopCompletionSucceeded),
        @"stop_completion_followed_request": @(self.stopCompletionFollowedRequest),
        @"stopped_status_observed": @(self.stoppedStatusObserved),
        @"stopped_status_followed_request": @(self.stoppedStatusFollowedRequest),
        @"delegate_stop_callback_observed": @(self.delegateStopCallbackObserved),
        @"delegate_stop_error_observed": @(self.delegateStopErrorObserved),
      },
      @"first_file_relative_pts_seconds": @(firstPTS),
      @"last_file_relative_pts_seconds": @(lastPTS),
      @"session_start_host_time": @{
        @"value": @(self.sessionStartHostPTS.value),
        @"timescale": @(self.sessionStartHostPTS.timescale),
      },
      @"health_gates": @{
        @"maximum_allowed_display_gap_seconds": @(MaximumDeliveryGapSeconds),
        @"maximum_allowed_terminal_recovery_gap_seconds":
            @(TerminalDeliveryGapRecoveryCeilingSeconds),
        @"maximum_allowed_callback_delivery_lag_seconds":
            @(MaximumCallbackDeliveryLagSeconds),
        @"maximum_allowed_callback_service_seconds": @(MaximumCallbackServiceSeconds),
        @"sample_accounting_passed": @(accountingPassed),
        @"terminal_gap_recovery_passed":
            jsonBoolean([self.terminalGapRecoveryMetadata[@"passed"] boolValue]),
        @"passed": jsonBoolean(accountingPassed &&
            [self.terminalGapRecoveryMetadata[@"passed"] boolValue] &&
            !self.failure),
      },
      @"post_write_encoded_audit": postWriteAudit,
    };
  });
  return result;
}
@end

static pid_t stoppedProcess = 0;

static BOOL parseSignedIntegerLine(NSString *line, NSString *prefix,
                                   int64_t *value) {
  if (![line hasPrefix:prefix]) return NO;
  NSString *tail = [line substringFromIndex:prefix.length];
  NSScanner *scanner = [NSScanner scannerWithString:tail];
  long long parsed = 0;
  if (![scanner scanLongLong:&parsed] || !scanner.isAtEnd) return NO;
  *value = (int64_t)parsed;
  return YES;
}

@interface PlaybackLogMonitor : NSObject
@property(nonatomic) int ptyMasterFD;
@property(nonatomic) int logFD;
@property(nonatomic) pid_t processID;
@property(nonatomic, copy) NSString *logPath;
@property(nonatomic, strong) NSCondition *condition;
@property(nonatomic, strong) NSMutableData *pendingLineBytes;
@property(nonatomic, strong) NSMutableArray<NSDictionary *> *stopEvents;
@property(nonatomic, strong) NSMutableString *generationFrameRows;
@property(nonatomic, copy) NSString *failureMessage;
@property(nonatomic, copy) NSString *expectedReplayPath;
@property(nonatomic, copy) NSString *seenReplayPath;
@property(nonatomic) int64_t requestedStartFrame;
@property(nonatomic) int64_t requestedInclusiveEndFrame;
@property(nonatomic) int64_t commandExclusiveEndFrame;
@property(nonatomic) int64_t seenStartFrame;
@property(nonatomic) int64_t seenEndFrame;
@property(nonatomic) int64_t seenGameEndFrame;
@property(nonatomic) int64_t lastGenerationFrame;
@property(nonatomic) int64_t stopTargetFrame;
@property(nonatomic) NSUInteger totalBytes;
@property(nonatomic) NSUInteger totalLines;
@property(nonatomic) NSUInteger generationArmByteOffset;
@property(nonatomic) NSUInteger generationArmLineOffset;
@property(nonatomic) NSUInteger generationArmPendingByteCount;
@property(nonatomic) NSUInteger fileMarkerLine;
@property(nonatomic) NSUInteger startMarkerLine;
@property(nonatomic) NSUInteger gameEndMarkerLine;
@property(nonatomic) NSUInteger endMarkerLine;
@property(nonatomic) NSUInteger preStartFrameLineCount;
@property(nonatomic) NSUInteger generationFrameCount;
@property(nonatomic) NSUInteger exclusiveSentinelLine;
@property(nonatomic) CFAbsoluteTime lastByteAt;
@property(nonatomic) BOOL readerStarted;
@property(nonatomic) BOOL readerEnded;
@property(nonatomic) BOOL generationArmed;
@property(nonatomic) BOOL fileMarkerSeen;
@property(nonatomic) BOOL startMarkerSeen;
@property(nonatomic) BOOL gameEndMarkerSeen;
@property(nonatomic) BOOL endMarkerSeen;
@property(nonatomic) BOOL generationMarkersComplete;
@property(nonatomic) BOOL stopTargetArmed;
@property(nonatomic) BOOL stopTargetReached;
@property(nonatomic) BOOL generationTraceComplete;
@property(nonatomic) BOOL exclusiveSentinelSeen;
- (instancetype)initWithPTYMasterFD:(int)ptyMasterFD
                            logPath:(NSString *)logPath
                          processID:(pid_t)processID;
- (void)start;
- (BOOL)waitUntilQuietForSeconds:(double)quietSeconds
                         timeout:(double)timeoutSeconds;
- (void)armGenerationForReplayPath:(NSString *)replayPath
                        startFrame:(int64_t)startFrame
               inclusiveEndFrame:(int64_t)inclusiveEndFrame
             exclusiveEndFrame:(int64_t)exclusiveEndFrame;
- (void)armStopAtFrame:(int64_t)frame;
- (BOOL)waitForArmedStopWithTimeout:(double)timeoutSeconds;
- (BOOL)stopTargetReachedSnapshot;
- (void)disarmStopTarget;
- (BOOL)waitForGenerationTraceWithTimeout:(double)timeoutSeconds;
- (NSError *)failureSnapshot;
- (NSDictionary *)metadataSnapshot;
@end

@implementation PlaybackLogMonitor

- (instancetype)initWithPTYMasterFD:(int)ptyMasterFD
                            logPath:(NSString *)logPath
                          processID:(pid_t)processID {
  if ((self = [super init])) {
    _ptyMasterFD = ptyMasterFD;
    _processID = processID;
    _logPath = [logPath copy];
    _condition = [NSCondition new];
    _pendingLineBytes = [NSMutableData data];
    _stopEvents = [NSMutableArray array];
    _generationFrameRows = [NSMutableString string];
    _seenStartFrame = INT64_MIN;
    _seenEndFrame = INT64_MIN;
    _seenGameEndFrame = INT64_MIN;
    _lastGenerationFrame = INT64_MIN;
    _stopTargetFrame = INT64_MIN;
    _lastByteAt = CFAbsoluteTimeGetCurrent();
    _logFD = open(logPath.fileSystemRepresentation,
                  O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (_logFD < 0) {
      _failureMessage = [NSString stringWithFormat:
          @"could not open Dolphin PTY log %@: %s", logPath, strerror(errno)];
    }
  }
  return self;
}

- (void)setFailureLocked:(NSString *)message stopProcess:(BOOL)stopProcess {
  if (self.failureMessage) return;
  self.failureMessage = message;
  if (stopProcess) {
    int result = kill(self.processID, SIGSTOP);
    [self.stopEvents addObject:@{
      @"kind": @"failure-stop",
      @"target_frame": @(self.stopTargetFrame),
      @"line_number": @(self.totalLines),
      @"signal_succeeded": jsonBoolean(result == 0),
      @"wall_time_seconds": @(CFAbsoluteTimeGetCurrent()),
    }];
  }
  [self.condition broadcast];
}

- (void)completeMarkersIfPossibleLocked {
  if (self.generationMarkersComplete || !self.fileMarkerSeen ||
      !self.startMarkerSeen || !self.gameEndMarkerSeen || !self.endMarkerSeen)
    return;
  NSString *actualPath = self.seenReplayPath.stringByStandardizingPath;
  NSString *expectedPath = self.expectedReplayPath.stringByStandardizingPath;
  if (![actualPath isEqualToString:expectedPath]) {
    [self setFailureLocked:[NSString stringWithFormat:
        @"capture generation replay marker mismatch: expected=%@ actual=%@",
        expectedPath, actualPath] stopProcess:YES];
    return;
  }
  if (self.seenStartFrame != self.requestedStartFrame ||
      self.seenEndFrame != self.commandExclusiveEndFrame) {
    [self setFailureLocked:[NSString stringWithFormat:
        @"capture generation frame markers mismatch: expected=%lld..%lld actual=%lld..%lld",
        self.requestedStartFrame, self.commandExclusiveEndFrame,
        self.seenStartFrame, self.seenEndFrame] stopProcess:YES];
    return;
  }
  if (self.seenGameEndFrame < self.requestedInclusiveEndFrame) {
    [self setFailureLocked:[NSString stringWithFormat:
        @"capture generation game-end marker precedes requested content: requested_end=%lld game_end=%lld",
        self.requestedInclusiveEndFrame, self.seenGameEndFrame]
               stopProcess:YES];
    return;
  }
  if (!(self.generationArmLineOffset < self.fileMarkerLine &&
        self.fileMarkerLine < self.startMarkerLine &&
        self.startMarkerLine < self.gameEndMarkerLine &&
        self.gameEndMarkerLine < self.endMarkerLine)) {
    [self setFailureLocked:@"capture generation markers are out of order"
               stopProcess:YES];
    return;
  }
  self.generationMarkersComplete = YES;
  [self.condition broadcast];
}

- (void)handleLineLocked:(NSString *)line {
  self.totalLines += 1;
  if (!self.generationArmed) return;

  NSString *filePrefix = @"[FILE_PATH] ";
  if ([line hasPrefix:filePrefix]) {
    if (self.generationMarkersComplete || self.fileMarkerSeen) {
      [self setFailureLocked:@"duplicate capture-generation replay-path marker"
                 stopProcess:YES];
      return;
    }
    self.seenReplayPath = [line substringFromIndex:filePrefix.length];
    self.fileMarkerSeen = YES;
    self.fileMarkerLine = self.totalLines;
    [self completeMarkersIfPossibleLocked];
    return;
  }

  int64_t value = 0;
  if (parseSignedIntegerLine(line, @"[PLAYBACK_START_FRAME] ", &value)) {
    if (self.generationMarkersComplete || self.startMarkerSeen) {
      [self setFailureLocked:@"duplicate capture-generation start marker"
                 stopProcess:YES];
      return;
    }
    self.seenStartFrame = value;
    self.startMarkerSeen = YES;
    self.startMarkerLine = self.totalLines;
    [self completeMarkersIfPossibleLocked];
    return;
  }
  if (parseSignedIntegerLine(line, @"[GAME_END_FRAME] ", &value)) {
    if (self.generationMarkersComplete || self.gameEndMarkerSeen) {
      [self setFailureLocked:@"duplicate capture-generation game-end marker"
                 stopProcess:YES];
      return;
    }
    self.seenGameEndFrame = value;
    self.gameEndMarkerSeen = YES;
    self.gameEndMarkerLine = self.totalLines;
    [self completeMarkersIfPossibleLocked];
    return;
  }
  if (parseSignedIntegerLine(line, @"[PLAYBACK_END_FRAME] ", &value)) {
    if (self.generationMarkersComplete || self.endMarkerSeen) {
      [self setFailureLocked:@"duplicate capture-generation end marker"
                 stopProcess:YES];
      return;
    }
    self.seenEndFrame = value;
    self.endMarkerSeen = YES;
    self.endMarkerLine = self.totalLines;
    [self completeMarkersIfPossibleLocked];
    return;
  }
  if (!parseSignedIntegerLine(line, @"[CURRENT_FRAME] ", &value) ||
      !self.generationMarkersComplete)
    return;

  if (value < self.requestedStartFrame) {
    self.preStartFrameLineCount += 1;
    return;
  }
  if (value == self.commandExclusiveEndFrame) {
    if (self.exclusiveSentinelSeen || !self.generationTraceComplete ||
        self.lastGenerationFrame != self.requestedInclusiveEndFrame) {
      [self setFailureLocked:
          @"exclusive CURRENT_FRAME sentinel arrived before the complete inclusive content trace"
               stopProcess:YES];
      return;
    }
    self.exclusiveSentinelSeen = YES;
    self.exclusiveSentinelLine = self.totalLines;
    [self.condition broadcast];
    return;
  }
  if (value > self.commandExclusiveEndFrame) {
    [self setFailureLocked:[NSString stringWithFormat:
        @"capture-generation frame exceeded exclusive command boundary: %lld",
        value]
             stopProcess:YES];
    return;
  }
  if (self.generationFrameCount == 0) {
    if (value != self.requestedStartFrame) {
      [self setFailureLocked:[NSString stringWithFormat:
          @"capture-generation first frame mismatch: expected=%lld actual=%lld",
          self.requestedStartFrame, value] stopProcess:YES];
      return;
    }
  } else if (value != self.lastGenerationFrame + 1) {
    [self setFailureLocked:[NSString stringWithFormat:
        @"capture-generation frame trace is not consecutive: previous=%lld actual=%lld",
        self.lastGenerationFrame, value] stopProcess:YES];
    return;
  }
  self.lastGenerationFrame = value;
  self.generationFrameCount += 1;
  [self.generationFrameRows appendFormat:@"%lld\n", value];

  if (self.stopTargetArmed) {
    if (value > self.stopTargetFrame) {
      [self setFailureLocked:[NSString stringWithFormat:
          @"playback startup stop overshot: target=%lld actual=%lld",
          self.stopTargetFrame, value] stopProcess:YES];
      return;
    }
    if (value == self.stopTargetFrame) {
      int result = kill(self.processID, SIGSTOP);
      [self.stopEvents addObject:@{
        @"kind": @"frame-stop",
        @"target_frame": @(self.stopTargetFrame),
        @"line_number": @(self.totalLines),
        @"generation_frame_count": @(self.generationFrameCount),
        @"signal_succeeded": jsonBoolean(result == 0),
        @"wall_time_seconds": @(CFAbsoluteTimeGetCurrent()),
      }];
      if (result != 0) {
        [self setFailureLocked:[NSString stringWithFormat:
            @"could not suspend Slippi Dolphin on frame %lld: %s",
            self.stopTargetFrame, strerror(errno)] stopProcess:NO];
        return;
      }
      self.stopTargetReached = YES;
      [self.condition broadcast];
    }
  }
  if (value == self.requestedInclusiveEndFrame) {
    self.generationTraceComplete = YES;
    [self.condition broadcast];
  }
}

- (void)start {
  [self.condition lock];
  if (self.readerStarted) {
    [self.condition unlock];
    return;
  }
  self.readerStarted = YES;
  [self.condition unlock];
  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    @autoreleasepool {
      uint8_t buffer[4096];
      while (YES) {
        ssize_t count = read(self.ptyMasterFD, buffer, sizeof(buffer));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
          [self.condition lock];
          self.readerEnded = YES;
          [self.condition broadcast];
          [self.condition unlock];
          return;
        }
        ssize_t written = 0;
        while (written < count) {
          ssize_t result = write(self.logFD, buffer + written,
                                 (size_t)(count - written));
          if (result < 0 && errno == EINTR) continue;
          if (result <= 0) {
            [self.condition lock];
            [self setFailureLocked:[NSString stringWithFormat:
                @"could not tee Dolphin PTY output: %s", strerror(errno)]
                         stopProcess:YES];
            [self.condition unlock];
            return;
          }
          written += result;
        }

        [self.condition lock];
        self.totalBytes += (NSUInteger)count;
        self.lastByteAt = CFAbsoluteTimeGetCurrent();
        [self.pendingLineBytes appendBytes:buffer length:(NSUInteger)count];
        while (YES) {
          const uint8_t *bytes = self.pendingLineBytes.bytes;
          NSUInteger length = self.pendingLineBytes.length;
          NSUInteger newline = NSNotFound;
          for (NSUInteger index = 0; index < length; ++index) {
            if (bytes[index] == '\n') {
              newline = index;
              break;
            }
          }
          if (newline == NSNotFound) break;
          NSUInteger lineLength = newline;
          if (lineLength > 0 && bytes[lineLength - 1] == '\r') lineLength -= 1;
          NSData *lineData = [self.pendingLineBytes subdataWithRange:
              NSMakeRange(0, lineLength)];
          NSString *line = [[NSString alloc] initWithData:lineData
                                                  encoding:NSUTF8StringEncoding];
          if (!line)
            line = [[NSString alloc] initWithData:lineData
                                          encoding:NSISOLatin1StringEncoding] ?: @"";
          [self.pendingLineBytes replaceBytesInRange:NSMakeRange(0, newline + 1)
                                           withBytes:NULL length:0];
          [self handleLineLocked:line];
        }
        [self.condition broadcast];
        [self.condition unlock];
      }
    }
  });
}

- (BOOL)waitUntilQuietForSeconds:(double)quietSeconds
                         timeout:(double)timeoutSeconds {
  NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeoutSeconds];
  [self.condition lock];
  while (!self.failureMessage) {
    double quiet = CFAbsoluteTimeGetCurrent() - self.lastByteAt;
    if (quiet >= quietSeconds) {
      if (self.logFD >= 0) fsync(self.logFD);
      [self.condition unlock];
      return YES;
    }
    if ([deadline timeIntervalSinceNow] <= 0.0) break;
    NSDate *wake = [NSDate dateWithTimeIntervalSinceNow:MIN(0.010, quietSeconds - quiet)];
    [self.condition waitUntilDate:wake];
  }
  [self.condition unlock];
  return NO;
}

- (void)armGenerationForReplayPath:(NSString *)replayPath
                        startFrame:(int64_t)startFrame
               inclusiveEndFrame:(int64_t)inclusiveEndFrame
             exclusiveEndFrame:(int64_t)exclusiveEndFrame {
  [self.condition lock];
  self.generationArmPendingByteCount = self.pendingLineBytes.length;
  if (self.generationArmPendingByteCount != 0) {
    [self setFailureLocked:
        @"Dolphin PTY has an unterminated idle line at the generation boundary"
               stopProcess:NO];
    [self.condition unlock];
    return;
  }
  self.expectedReplayPath = [replayPath copy];
  self.requestedStartFrame = startFrame;
  self.requestedInclusiveEndFrame = inclusiveEndFrame;
  self.commandExclusiveEndFrame = exclusiveEndFrame;
  self.generationArmByteOffset = self.totalBytes;
  self.generationArmLineOffset = self.totalLines;
  self.generationArmed = YES;
  [self.condition unlock];
}

- (void)armStopAtFrame:(int64_t)frame {
  [self.condition lock];
  if (self.failureMessage) {
    [self.condition unlock];
    return;
  }
  self.stopTargetFrame = frame;
  self.stopTargetArmed = YES;
  self.stopTargetReached = NO;
  [self.condition unlock];
}

- (BOOL)waitForArmedStopWithTimeout:(double)timeoutSeconds {
  NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeoutSeconds];
  [self.condition lock];
  while (!self.stopTargetReached && !self.failureMessage &&
         !self.readerEnded && [deadline timeIntervalSinceNow] > 0.0)
    [self.condition waitUntilDate:deadline];
  BOOL result = self.stopTargetReached && !self.failureMessage;
  [self.condition unlock];
  return result;
}

- (BOOL)stopTargetReachedSnapshot {
  [self.condition lock];
  BOOL result = self.stopTargetReached && !self.failureMessage;
  [self.condition unlock];
  return result;
}

- (void)disarmStopTarget {
  [self.condition lock];
  self.stopTargetArmed = NO;
  self.stopTargetReached = NO;
  self.stopTargetFrame = INT64_MIN;
  [self.condition unlock];
}

- (BOOL)waitForGenerationTraceWithTimeout:(double)timeoutSeconds {
  NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeoutSeconds];
  [self.condition lock];
  while (!self.generationTraceComplete && !self.failureMessage &&
         !self.readerEnded && [deadline timeIntervalSinceNow] > 0.0)
    [self.condition waitUntilDate:deadline];
  BOOL expectedCount = self.generationFrameCount ==
      (NSUInteger)(self.requestedInclusiveEndFrame - self.requestedStartFrame + 1);
  BOOL result = self.generationTraceComplete && expectedCount &&
      self.lastGenerationFrame == self.requestedInclusiveEndFrame &&
      !self.failureMessage;
  if (!result && !self.failureMessage) {
    self.failureMessage = [NSString stringWithFormat:
        @"capture-generation trace incomplete: expected=%lld..%lld count=%llu actual_last=%lld actual_count=%llu",
        self.requestedStartFrame, self.requestedInclusiveEndFrame,
        (unsigned long long)(self.requestedInclusiveEndFrame - self.requestedStartFrame + 1),
        self.lastGenerationFrame, (unsigned long long)self.generationFrameCount];
  }
  [self.condition unlock];
  return result;
}

- (NSError *)failureSnapshot {
  [self.condition lock];
  NSString *message = [self.failureMessage copy];
  [self.condition unlock];
  return message ? captureError(message) : nil;
}

- (NSDictionary *)metadataSnapshot {
  [self.condition lock];
  if (self.logFD >= 0) fsync(self.logFD);
  NSString *rows = [self.generationFrameRows copy];
  NSData *rowData = [rows dataUsingEncoding:NSUTF8StringEncoding];
  NSDictionary *result = @{
    @"schema": @"idle-command-zero-audio-two-stop-inclusive-terminal-v7",
    @"pty_master_fd": @(self.ptyMasterFD),
    @"log_path": self.logPath,
    @"total_pty_bytes": @(self.totalBytes),
    @"total_pty_lines": @(self.totalLines),
    @"generation_arm_byte_offset": @(self.generationArmByteOffset),
    @"generation_arm_line_offset": @(self.generationArmLineOffset),
    @"generation_arm_pending_byte_count":
        @(self.generationArmPendingByteCount),
    @"expected_replay_path": self.expectedReplayPath ?: @"",
    @"seen_replay_path": self.seenReplayPath ?: @"",
    @"requested_start_frame": @(self.requestedStartFrame),
    @"requested_inclusive_end": @(self.requestedInclusiveEndFrame),
    @"command_exclusive_end": @(self.commandExclusiveEndFrame),
    @"command_boundary_semantics": @"endFrame-is-exclusive-control-boundary",
    @"generation_boundary_semantics":
        @"CURRENT_FRAME-ends-at-requested-inclusive-last-content-frame",
    @"seen_start_frame": @(self.seenStartFrame),
    @"seen_end_frame": @(self.seenEndFrame),
    @"seen_game_end_frame": @(self.seenGameEndFrame),
    @"file_marker_line": @(self.fileMarkerLine),
    @"start_marker_line": @(self.startMarkerLine),
    @"game_end_marker_line": @(self.gameEndMarkerLine),
    @"end_marker_line": @(self.endMarkerLine),
    @"pre_start_frame_line_count": @(self.preStartFrameLineCount),
    @"generation_frame_count": @(self.generationFrameCount),
    @"generation_first_frame": self.generationFrameCount > 0 ?
        @(self.requestedStartFrame) : [NSNull null],
    @"generation_last_frame": self.generationFrameCount > 0 ?
        @(self.lastGenerationFrame) : [NSNull null],
    @"exclusive_sentinel_observed": jsonBoolean(self.exclusiveSentinelSeen),
    @"exclusive_sentinel_line": @(self.exclusiveSentinelLine),
    @"exclusive_sentinel_excluded_from_content_trace": @YES,
    @"generation_frame_trace": @{
      @"encoding": @"signed-frame-number-newline-v1",
      @"row_count": @(self.generationFrameCount),
      @"rows": rows,
      @"sha256": sha256Hex(rowData),
    },
    @"stop_events": [self.stopEvents copy],
    @"markers_complete": @(self.generationMarkersComplete),
    @"trace_complete": @(self.generationTraceComplete),
    @"reader_ended": @(self.readerEnded),
    @"failure": self.failureMessage ?: @"",
    @"passed": jsonBoolean(!self.failureMessage && self.generationMarkersComplete &&
                            self.generationTraceComplete),
  };
  [self.condition unlock];
  return result;
}
@end

static void fail(NSString *message) {
  if (stoppedProcess > 0) {
    kill(stoppedProcess, SIGCONT);
    stoppedProcess = 0;
  }
  fprintf(stderr, "%s\n", message.UTF8String);
  exit(1);
}

static BOOL writeAll(int fd, const uint8_t *bytes, NSUInteger length) {
  NSUInteger offset = 0;
  while (offset < length) {
    ssize_t count = write(fd, bytes + offset, length - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return NO;
    offset += (NSUInteger)count;
  }
  return YES;
}

static NSDictionary *installCapturePlaybackCommand(
    NSString *activePath, NSString *captureTemplatePath,
    int64_t requestedStartFrame, int64_t requestedInclusiveEndFrame) {
  if (requestedStartFrame < INT32_MIN || requestedStartFrame > INT32_MAX ||
      requestedInclusiveEndFrame < requestedStartFrame ||
      requestedInclusiveEndFrame >= INT32_MAX)
    fail(@"requested replay frame interval cannot supply an exclusive end sentinel");
  int64_t commandExclusiveEndFrame = requestedInclusiveEndFrame + 1;

  struct stat activeStatus;
  if (lstat(activePath.fileSystemRepresentation, &activeStatus) != 0 ||
      !S_ISREG(activeStatus.st_mode))
    fail([NSString stringWithFormat:@"active playback command is not a regular file: %@",
                                    activePath]);
  struct stat templateStatus;
  if (lstat(captureTemplatePath.fileSystemRepresentation, &templateStatus) != 0 ||
      !S_ISREG(templateStatus.st_mode))
    fail([NSString stringWithFormat:@"capture playback command is not a regular file: %@",
                                    captureTemplatePath]);
  NSData *activeData = [NSData dataWithContentsOfFile:activePath];
  NSData *templateData = [NSData dataWithContentsOfFile:captureTemplatePath];
  if (!activeData || !templateData)
    fail(@"could not read idle and capture playback commands");
  NSError *jsonError = nil;
  id activeObject = [NSJSONSerialization JSONObjectWithData:activeData options:0
                                                       error:&jsonError];
  if (![activeObject isKindOfClass:NSDictionary.class])
    fail([NSString stringWithFormat:@"idle playback command is invalid: %@", jsonError]);
  NSDictionary *activePayload = activeObject;
  if (![activePayload[@"mode"] isEqual:@"normal"] ||
      ![activePayload[@"replay"] isKindOfClass:NSString.class] ||
      [activePayload[@"replay"] length] != 0)
    fail(@"active playback command must be an idle normal command with an empty replay");

  jsonError = nil;
  id templateObject = [NSJSONSerialization JSONObjectWithData:templateData
                                                       options:NSJSONReadingMutableContainers
                                                         error:&jsonError];
  if (![templateObject isKindOfClass:NSMutableDictionary.class])
    fail([NSString stringWithFormat:@"capture playback command is invalid: %@", jsonError]);
  NSMutableDictionary *capturePayload = templateObject;
  NSString *replayPath = capturePayload[@"replay"];
  if (![capturePayload[@"mode"] isEqual:@"normal"] ||
      ![replayPath isKindOfClass:NSString.class] || replayPath.length == 0 ||
      ![capturePayload[@"isRealTimeMode"] isEqual:@NO] ||
      ![capturePayload[@"shouldResync"] isEqual:@NO] ||
      ![capturePayload[@"rollbackDisplayMethod"] isEqual:@"off"])
    fail(@"capture playback command does not use the required deterministic normal-playback settings");
  NSString *captureCommandID = [NSString stringWithFormat:
      @"game-video-capture-%d-%@", getpid(), NSUUID.UUID.UUIDString.lowercaseString];
  capturePayload[@"startFrame"] = @(requestedStartFrame);
  capturePayload[@"endFrame"] = @(commandExclusiveEndFrame);
  capturePayload[@"commandId"] = captureCommandID;
  NSData *installedData = [NSJSONSerialization dataWithJSONObject:capturePayload
      options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:&jsonError];
  if (!installedData)
    fail([NSString stringWithFormat:@"could not encode capture playback command: %@",
                                    jsonError]);
  NSMutableData *terminatedData = [installedData mutableCopy];
  const uint8_t newline = '\n';
  [terminatedData appendBytes:&newline length:1];

  NSString *directory = activePath.stringByDeletingLastPathComponent;
  NSString *temporaryPath = [directory stringByAppendingPathComponent:
      [NSString stringWithFormat:@".%@.%@.tmp", activePath.lastPathComponent,
                                 NSUUID.UUID.UUIDString.lowercaseString]];
  int temporaryFD = open(temporaryPath.fileSystemRepresentation,
                         O_WRONLY | O_CREAT | O_EXCL, activeStatus.st_mode & 0777);
  if (temporaryFD < 0)
    fail([NSString stringWithFormat:@"could not create atomic playback command: %s",
                                    strerror(errno)]);
  if (!writeAll(temporaryFD, terminatedData.bytes, terminatedData.length)) {
    int savedErrno = errno;
    close(temporaryFD);
    unlink(temporaryPath.fileSystemRepresentation);
    fail([NSString stringWithFormat:@"could not write atomic playback command: %s",
                                    strerror(savedErrno)]);
  }
  time_t installedMTime = activeStatus.st_mtime <= (time_t)(LLONG_MAX - 2) ?
      activeStatus.st_mtime + 2 : activeStatus.st_mtime - 2;
  struct timeval times[2] = {
    {.tv_sec = installedMTime, .tv_usec = 0},
    {.tv_sec = installedMTime, .tv_usec = 0},
  };
  if (futimes(temporaryFD, times) != 0 || fsync(temporaryFD) != 0) {
    int savedErrno = errno;
    close(temporaryFD);
    unlink(temporaryPath.fileSystemRepresentation);
    fail([NSString stringWithFormat:@"could not seal atomic playback command: %s",
                                    strerror(savedErrno)]);
  }
  if (close(temporaryFD) != 0) {
    int savedErrno = errno;
    unlink(temporaryPath.fileSystemRepresentation);
    fail([NSString stringWithFormat:@"could not close atomic playback command: %s",
                                    strerror(savedErrno)]);
  }
  if (rename(temporaryPath.fileSystemRepresentation,
             activePath.fileSystemRepresentation) != 0) {
    int savedErrno = errno;
    unlink(temporaryPath.fileSystemRepresentation);
    fail([NSString stringWithFormat:@"could not install atomic playback command: %s",
                                    strerror(savedErrno)]);
  }
  int directoryFD = open(directory.fileSystemRepresentation, O_RDONLY);
  if (directoryFD < 0 || fsync(directoryFD) != 0) {
    int savedErrno = errno;
    if (directoryFD >= 0) close(directoryFD);
    fail([NSString stringWithFormat:@"could not seal playback command directory: %s",
                                    strerror(savedErrno)]);
  }
  close(directoryFD);

  struct stat installedStatus;
  NSData *verifiedData = [NSData dataWithContentsOfFile:activePath];
  if (lstat(activePath.fileSystemRepresentation, &installedStatus) != 0 ||
      !S_ISREG(installedStatus.st_mode) ||
      installedStatus.st_mtime == activeStatus.st_mtime ||
      installedStatus.st_mtime != installedMTime ||
      installedStatus.st_mtimespec.tv_nsec != 0 ||
      ![verifiedData isEqualToData:terminatedData])
    fail(@"atomic capture playback command identity verification failed");
  return @{
    @"method": @"same-directory-rename-distinct-whole-second-mtime-v1",
    @"active_path": activePath,
    @"capture_template_path": captureTemplatePath,
    @"idle_command_sha256": sha256Hex(activeData),
    @"capture_template_sha256": sha256Hex(templateData),
    @"installed_command_sha256": sha256Hex(terminatedData),
    @"idle_command_mtime_seconds": @((long long)activeStatus.st_mtime),
    @"idle_command_mtime_nanoseconds": @((long long)activeStatus.st_mtimespec.tv_nsec),
    @"installed_command_mtime_seconds": @((long long)installedStatus.st_mtime),
    @"installed_command_mtime_nanoseconds":
        @((long long)installedStatus.st_mtimespec.tv_nsec),
    @"mtime_seconds_distinct":
        jsonBoolean(installedStatus.st_mtime != activeStatus.st_mtime),
    @"capture_command_id": captureCommandID,
    @"replay_path": replayPath,
    @"requested_start_frame": @(requestedStartFrame),
    @"requested_inclusive_end": @(requestedInclusiveEndFrame),
    @"command_exclusive_end": @(commandExclusiveEndFrame),
    @"passed": @YES,
  };
}

static SCShareableContent *shareableContent(void) {
  __block SCShareableContent *content = nil;
  __block NSError *failure = nil;
  dispatch_semaphore_t ready = dispatch_semaphore_create(0);
  [SCShareableContent
      getShareableContentExcludingDesktopWindows:YES
                             onScreenWindowsOnly:NO
                                completionHandler:^(SCShareableContent *value, NSError *error) {
                                  content = value;
                                  failure = error;
                                  dispatch_semaphore_signal(ready);
                                }];
  dispatch_semaphore_wait(ready, DISPATCH_TIME_FOREVER);
  if (failure) fail([NSString stringWithFormat:@"could not list windows: %@", failure]);
  return content;
}

static NSDictionary<NSNumber *, NSDictionary *> *layerZeroCoreGraphicsWindows(
    pid_t processID) {
  NSArray *windows = CFBridgingRelease(CGWindowListCopyWindowInfo(
      kCGWindowListOptionAll | kCGWindowListExcludeDesktopElements, kCGNullWindowID));
  NSMutableDictionary<NSNumber *, NSDictionary *> *result = [NSMutableDictionary dictionary];
  for (NSDictionary *window in windows) {
    if ([window[(id)kCGWindowOwnerPID] intValue] != processID ||
        [window[(id)kCGWindowLayer] intValue] != 0)
      continue;
    CGRect frame = CGRectZero;
    NSDictionary *bounds = window[(id)kCGWindowBounds];
    CGRectMakeWithDictionaryRepresentation((__bridge CFDictionaryRef)bounds, &frame);
    NSNumber *windowID = window[(id)kCGWindowNumber];
    if (!windowID) continue;
    result[windowID] = @{
      @"title": window[(id)kCGWindowName] ?: @"",
      @"core_width": @(frame.size.width),
      @"core_height": @(frame.size.height),
    };
  }
  return result;
}

static SCWindow *waitForLargestWindow(pid_t processID, double timeoutSeconds,
                                      size_t minimumPixelWidth,
                                      size_t minimumPixelHeight,
                                      size_t expectedPixelWidth,
                                      size_t expectedPixelHeight) {
  NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeoutSeconds];
  CGWindowID stableWindow = 0;
  size_t stablePixelWidth = 0;
  size_t stablePixelHeight = 0;
  int stablePolls = 0;
  NSMutableSet<NSString *> *loggedRejections = [NSMutableSet set];
  while ([deadline timeIntervalSinceNow] > 0.0) {
    NSDictionary<NSNumber *, NSDictionary *> *coreWindows =
        layerZeroCoreGraphicsWindows(processID);
    NSMutableArray<NSDictionary *> *eligible = [NSMutableArray array];
    for (SCWindow *window in shareableContent().windows) {
      NSDictionary *coreWindow = coreWindows[@(window.windowID)];
      if (!coreWindow) continue;
      SCContentFilter *filter =
          [[SCContentFilter alloc] initWithDesktopIndependentWindow:window];
      SCShareableContentInfo *info = [SCShareableContent infoForFilter:filter];
      double scale = MAX(1.0, info.pointPixelScale);
      size_t pixelWidth = (size_t)llround(window.frame.size.width * scale);
      size_t pixelHeight = (size_t)llround(window.frame.size.height * scale);
      pixelWidth -= pixelWidth % 2;
      pixelHeight -= pixelHeight % 2;
      BOOL meetsMinimum =
          pixelWidth >= minimumPixelWidth && pixelHeight >= minimumPixelHeight;
      BOOL meetsExpected = expectedPixelWidth == 0 ||
          (pixelWidth == expectedPixelWidth && pixelHeight == expectedPixelHeight);
      if (meetsMinimum && meetsExpected) {
        [eligible addObject:@{
          @"window": window,
          @"pixel_width": @(pixelWidth),
          @"pixel_height": @(pixelHeight),
        }];
        continue;
      }
      NSString *title = coreWindow[@"title"];
      NSString *rejection = [NSString stringWithFormat:@"%u:%@:%zux%zu",
                                                       window.windowID, title,
                                                       pixelWidth, pixelHeight];
      if (![loggedRejections containsObject:rejection]) {
        fprintf(stderr,
                "rejected Dolphin layer-0 window=%u title=%s core=%.0fx%.0f sck=%zux%zu minimum=%zux%zu expected=%zux%zu\n",
                window.windowID, title.UTF8String,
                [coreWindow[@"core_width"] doubleValue],
                [coreWindow[@"core_height"] doubleValue], pixelWidth, pixelHeight,
                minimumPixelWidth, minimumPixelHeight, expectedPixelWidth,
                expectedPixelHeight);
        [loggedRejections addObject:rejection];
      }
    }

    NSDictionary *selected = nil;
    if (expectedPixelWidth > 0) {
      if (eligible.count == 1) selected = eligible.firstObject;
    } else {
      for (NSDictionary *candidate in eligible) {
        if (!selected ||
            [candidate[@"pixel_width"] unsignedLongLongValue] *
                [candidate[@"pixel_height"] unsignedLongLongValue] >
            [selected[@"pixel_width"] unsignedLongLongValue] *
                [selected[@"pixel_height"] unsignedLongLongValue])
          selected = candidate;
      }
    }
    if (selected) {
      SCWindow *window = selected[@"window"];
      size_t pixelWidth = [selected[@"pixel_width"] unsignedLongLongValue];
      size_t pixelHeight = [selected[@"pixel_height"] unsignedLongLongValue];
      if (window.windowID == stableWindow && pixelWidth == stablePixelWidth &&
          pixelHeight == stablePixelHeight) {
        stablePolls += 1;
      } else {
        stableWindow = window.windowID;
        stablePixelWidth = pixelWidth;
        stablePixelHeight = pixelHeight;
        stablePolls = 1;
      }
      if (stablePolls >= 4) return window;
    } else {
      stableWindow = 0;
      stablePixelWidth = 0;
      stablePixelHeight = 0;
      stablePolls = 0;
    }
    [NSThread sleepForTimeInterval:0.05];
  }
  return nil;
}

static uint16_t readLittleEndian16(const uint8_t *bytes) {
  return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
}

static uint32_t readLittleEndian32(const uint8_t *bytes) {
  return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
         ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

typedef struct {
  BOOL valid;
  uint16_t blockAlign;
  uint32_t sampleRate;
  NSUInteger dataOffset;
} WaveLayout;

// Resolve the stable layout of a growing RIFF/WAVE file once. Dolphin may not
// finalize the data-chunk length until shutdown, so later clock reads use the
// current file length rather than the declared data length.
static WaveLayout waveLayoutAtPath(NSString *path) {
  WaveLayout layout = {NO, 0, 0, 0};
  NSData *data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:nil];
  if (!data || data.length < 12) return layout;
  const uint8_t *bytes = data.bytes;
  if (memcmp(bytes, "RIFF", 4) != 0 || memcmp(bytes + 8, "WAVE", 4) != 0) return layout;
  uint16_t blockAlign = 0;
  uint32_t sampleRate = 0;
  NSUInteger offset = 12;
  while (offset + 8 <= data.length) {
    const uint8_t *chunk = bytes + offset;
    uint32_t declaredLength = readLittleEndian32(chunk + 4);
    NSUInteger payload = offset + 8;
    if (memcmp(chunk, "fmt ", 4) == 0) {
      if (declaredLength < 16 || payload + 16 > data.length) return layout;
      sampleRate = readLittleEndian32(bytes + payload + 4);
      blockAlign = readLittleEndian16(bytes + payload + 12);
      if (blockAlign == 0 || sampleRate == 0) return layout;
    } else if (memcmp(chunk, "data", 4) == 0) {
      if (blockAlign == 0 || sampleRate == 0 || payload > data.length) return layout;
      layout.valid = YES;
      layout.blockAlign = blockAlign;
      layout.sampleRate = sampleRate;
      layout.dataOffset = payload;
      return layout;
    }
    NSUInteger paddedLength = (NSUInteger)declaredLength + (declaredLength & 1U);
    if (paddedLength > data.length - payload) return layout;
    offset = payload + paddedLength;
  }
  return layout;
}

static int64_t waveFramesAtPath(NSString *path, WaveLayout layout) {
  if (!layout.valid) return -1;
  struct stat status;
  if (stat(path.fileSystemRepresentation, &status) != 0 || status.st_size < 0) return -1;
  uint64_t size = (uint64_t)status.st_size;
  if (size < layout.dataOffset) return -1;
  return (int64_t)((size - layout.dataOffset) / layout.blockAlign);
}

// The idle playback command cannot feed Dolphin's WaveFile writers. Before
// arming capture, accept either the empirically observed empty files or a
// complete WAVE header whose data payload still contains exactly zero frames.
// Any partial header or visible PCM is an ambiguous pre-boundary state and is
// rejected.
static NSDictionary *zeroAudioBoundaryState(NSString *path) {
  struct stat status;
  BOOL exists = stat(path.fileSystemRepresentation, &status) == 0;
  BOOL regularFile = exists && S_ISREG(status.st_mode);
  uint64_t size = regularFile && status.st_size >= 0 ? (uint64_t)status.st_size : 0;
  WaveLayout layout = waveLayoutAtPath(path);
  int64_t frames = layout.valid ? waveFramesAtPath(path, layout) : -1;
  BOOL emptyFile = regularFile && size == 0;
  BOOL headerOnly = regularFile && layout.valid && frames == 0 &&
      size == (uint64_t)layout.dataOffset;
  BOOL passed = emptyFile || headerOnly;
  return @{
    @"path": path,
    @"exists": @(exists),
    @"regular_file": @(regularFile),
    @"size_bytes": @(size),
    @"layout_valid": @(layout.valid),
    @"sample_rate": @(layout.sampleRate),
    @"block_align": @(layout.blockAlign),
    @"data_offset_bytes": @(layout.dataOffset),
    @"data_frames": @(frames),
    @"logical_data_frames": passed ? @0 : [NSNull null],
    @"empty_file": @(emptyFile),
    @"header_only_zero_data": @(headerOnly),
    @"zero_audio_passed": @(passed),
  };
}

static void appendClockLandmark(NSMutableArray<NSDictionary *> *landmarks,
                                double sourceVideoSeconds,
                                double audioSeconds) {
  if (!isfinite(sourceVideoSeconds) || !isfinite(audioSeconds) ||
      sourceVideoSeconds < 0.0 || audioSeconds < 0.0)
    fail(@"invalid replay clock landmark");
  NSDictionary *previous = landmarks.lastObject;
  if (previous) {
    double previousSource = [previous[@"source_video_seconds"] doubleValue];
    double previousAudio = [previous[@"audio_seconds"] doubleValue];
    if (sourceVideoSeconds <= previousSource || audioSeconds <= previousAudio)
      fail(@"replay clock landmarks must increase strictly");
  }
  [landmarks addObject:@{
    @"source_video_seconds": @(sourceVideoSeconds),
    @"audio_seconds": @(audioSeconds),
  }];
}

static NSDictionary *auditEncodedVideoSamples(NSURL *url,
                                              NSArray<NSNumber *> *expectedPTSSequence,
                                              double strictStartSeconds) {
  NSUInteger expectedSampleCount = expectedPTSSequence.count;
  AVURLAsset *asset = [AVURLAsset URLAssetWithURL:url options:nil];
  AVAssetTrack *videoTrack = [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
  if (!videoTrack) fail(@"post-write audit found no encoded video track");
  NSError *readerError = nil;
  AVAssetReader *reader = [AVAssetReader assetReaderWithAsset:asset error:&readerError];
  if (!reader) fail([NSString stringWithFormat:@"could not create post-write video reader: %@",
                                               readerError]);
  AVAssetReaderTrackOutput *output =
      [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:nil];
  output.alwaysCopiesSampleData = NO;
  if (![reader canAddOutput:output]) fail(@"could not attach post-write encoded video audit");
  [reader addOutput:output];
  if (![reader startReading])
    fail([NSString stringWithFormat:@"could not start post-write encoded video audit: %@",
                                    reader.error]);

  NSUInteger sampleCount = 0;
  NSUInteger readerBufferCount = 0;
  NSUInteger markerBufferCount = 0;
  double firstPTS = NAN;
  double lastPTS = NAN;
  double maximumPTSGap = 0.0;
  double maximumStrictPTSGap = 0.0;
  double maximumPTSSequenceError = 0.0;
  double maximumPTSGapReconciliationError = 0.0;
  CMSampleBufferRef sample = NULL;
  while ((sample = [output copyNextSampleBuffer])) {
    readerBufferCount += 1;
    CMItemCount samplesInBuffer = CMSampleBufferGetNumSamples(sample);
    if (samplesInBuffer == 0) {
      BOOL markerValid = CMSampleBufferIsValid(sample);
      size_t totalSampleSize = markerValid ?
          CMSampleBufferGetTotalSampleSize(sample) : SIZE_MAX;
      if (!markerValid || totalSampleSize != 0) {
        CFRelease(sample);
        fail(@"post-write encoded marker buffer is invalid or contains media bytes");
      }
      markerBufferCount += 1;
      CFRelease(sample);
      continue;
    }
    if (samplesInBuffer != 1) {
      CFRelease(sample);
      fail([NSString stringWithFormat:
          @"post-write encoded buffer contains %ld samples, expected one",
          (long)samplesInBuffer]);
    }
    CMTime pts = CMSampleBufferGetPresentationTimeStamp(sample);
    double seconds = CMTimeGetSeconds(pts);
    if (!CMTIME_IS_NUMERIC(pts) || !isfinite(seconds)) {
      CFRelease(sample);
      fail(@"post-write audit found an invalid encoded video PTS");
    }
    if (sampleCount >= expectedSampleCount) {
      CFRelease(sample);
      fail(@"post-write encoded video contains an unexpected extra sample");
    }
    double expectedPTS = expectedPTSSequence[sampleCount].doubleValue;
    double sequenceError = fabs(seconds - expectedPTS);
    maximumPTSSequenceError = MAX(maximumPTSSequenceError, sequenceError);
    if (sequenceError >
        EncodedPTSSequenceErrorSeconds + EncodedPTSNumericalSlackSeconds) {
      CFRelease(sample);
      fail([NSString stringWithFormat:
          @"post-write encoded PTS sequence mismatch at sample %zu: expected=%.9f actual=%.9f",
          sampleCount, expectedPTS, seconds]);
    }
    if (sampleCount == 0) {
      firstPTS = seconds;
    } else {
      double gap = seconds - lastPTS;
      if (!isfinite(gap) || gap <= 0.0) {
        CFRelease(sample);
        fail(@"post-write encoded video PTS values must increase strictly");
      }
      double priorExpectedPTS = expectedPTSSequence[sampleCount - 1].doubleValue;
      double roundedExpectedGap =
          (double)(llround(expectedPTS * 1000000.0) -
                   llround(priorExpectedPTS * 1000000.0)) /
          1000000.0;
      double gapReconciliationError = fabs(gap - roundedExpectedGap);
      maximumPTSGapReconciliationError =
          MAX(maximumPTSGapReconciliationError, gapReconciliationError);
      if (gapReconciliationError > encodedPTSGapReconciliationErrorSeconds() +
                                       EncodedPTSNumericalSlackSeconds) {
        CFRelease(sample);
        fail([NSString stringWithFormat:
            @"post-write encoded PTS gap disagrees with its rounded appended trace at sample %zu: trace=%.9f encoded=%.9f error=%.9f",
            sampleCount, roundedExpectedGap, gap, gapReconciliationError]);
      }
      maximumPTSGap = MAX(maximumPTSGap, gap);
      if (expectedPTS > strictStartSeconds)
        maximumStrictPTSGap = MAX(maximumStrictPTSGap, gap);
    }
    lastPTS = seconds;
    sampleCount += (NSUInteger)samplesInBuffer;
    CFRelease(sample);
  }
  if (reader.status != AVAssetReaderStatusCompleted)
    fail([NSString stringWithFormat:@"post-write encoded video audit failed: %@",
                                    reader.error]);
  if (sampleCount != expectedSampleCount)
    fail([NSString stringWithFormat:
        @"post-write encoded sample count mismatch: expected=%zu actual=%zu",
        expectedSampleCount, sampleCount]);
  if (sampleCount == 0 || fabs(firstPTS) > 0.001)
    fail([NSString stringWithFormat:@"post-write encoded video has wrong first PTS: %.9f",
                                    firstPTS]);
  if (maximumStrictPTSGap >
      maximumEncodedPTSGapSeconds() + EncodedPTSNumericalSlackSeconds)
    fail([NSString stringWithFormat:
        @"post-write encoded PTS gap %.6f exceeds %.6f seconds",
        maximumStrictPTSGap, MaximumDeliveryGapSeconds]);
  return @{
    @"method": @"avassetreader-encoded-media-samples-zero-size-markers-v2",
    @"expected_sample_count": @(expectedSampleCount),
    @"encoded_sample_count": @(sampleCount),
    @"reader_buffer_count": @(readerBufferCount),
    @"marker_buffer_count": @(markerBufferCount),
    @"first_pts_seconds": @(firstPTS),
    @"last_pts_seconds": @(lastPTS),
    @"maximum_pts_gap_seconds": @(maximumPTSGap),
    @"maximum_strict_pts_gap_seconds": @(maximumStrictPTSGap),
    @"strict_start_source_seconds": @(strictStartSeconds),
    @"maximum_pts_sequence_error_seconds": @(maximumPTSSequenceError),
    @"maximum_pts_gap_reconciliation_error_seconds":
        @(maximumPTSGapReconciliationError),
    @"maximum_allowed_pts_sequence_error_seconds": @(EncodedPTSSequenceErrorSeconds),
    @"pts_trace_rounding_seconds": @(EncodedPTSTraceRoundingSeconds),
    @"maximum_allowed_pts_gap_reconciliation_error_seconds":
        @(encodedPTSGapReconciliationErrorSeconds()),
    @"maximum_allowed_pts_gap_seconds": @(maximumEncodedPTSGapSeconds()),
    @"sample_count_matched": @YES,
    @"pts_strictly_increasing": @YES,
    @"pts_sequence_matched": @YES,
    @"pts_gap_reconciled": @YES,
    @"marker_buffers_validated": @YES,
    @"passed": @YES,
  };
}

static void activatePlaybackApplication(pid_t processID, double timeoutSeconds) {
  NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:MIN(timeoutSeconds, 5.0)];
  while ([deadline timeIntervalSinceNow] > 0.0) {
    NSRunningApplication *application =
        [NSRunningApplication runningApplicationWithProcessIdentifier:processID];
    if (application) {
      BOOL activated =
          [application activateWithOptions:NSApplicationActivateIgnoringOtherApps];
      fprintf(stderr, "activated Slippi Dolphin pid=%d result=%d\n", processID,
              activated);
      return;
    }
    [NSThread sleepForTimeInterval:0.020];
  }
  fail([NSString stringWithFormat:@"could not find Slippi Dolphin application pid %d",
                                  processID]);
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    [NSApplication sharedApplication];
    if (argc != 22) {
      fprintf(stderr,
              "usage: %s DOLPHIN_PID STARTUP_TIMEOUT GAME_SECONDS TAIL_SECONDS "
              "RESUME_DELAY DSP_AUDIO DTK_AUDIO MIN_WIDTH MIN_HEIGHT "
              "EXPECTED_WIDTH EXPECTED_HEIGHT REQUESTED_CAPTURE_FPS "
              "PLAYBACK_EMULATION_SPEED AUDIO_PRESENTATION_DELAY OUTPUT.mp4 PTY_MASTER_FD "
              "ACTIVE_PLAYBACK_JSON CAPTURE_PLAYBACK_JSON DOLPHIN_LOG "
              "REQUESTED_START_FRAME REQUESTED_INCLUSIVE_END_FRAME\n",
              argv[0]);
      return 2;
    }
    pid_t processID = (pid_t)strtol(argv[1], NULL, 10);
    double startupTimeout = strtod(argv[2], NULL);
    double gameSeconds = strtod(argv[3], NULL);
    double tailSeconds = strtod(argv[4], NULL);
    double resumeDelay = strtod(argv[5], NULL);
    NSString *dspAudioPath = @(argv[6]);
    NSString *dtkAudioPath = @(argv[7]);
    size_t minimumPixelWidth = (size_t)strtoull(argv[8], NULL, 10);
    size_t minimumPixelHeight = (size_t)strtoull(argv[9], NULL, 10);
    size_t expectedPixelWidth = (size_t)strtoull(argv[10], NULL, 10);
    size_t expectedPixelHeight = (size_t)strtoull(argv[11], NULL, 10);
    int requestedCaptureFPS = (int)strtol(argv[12], NULL, 10);
    double playbackEmulationSpeed = strtod(argv[13], NULL);
    double audioPresentationDelay = strtod(argv[14], NULL);
    NSURL *outputURL = [NSURL fileURLWithPath:@(argv[15])];
    int ptyMasterFD = (int)strtol(argv[16], NULL, 10);
    NSString *activePlaybackPath = @(argv[17]);
    NSString *capturePlaybackPath = @(argv[18]);
    NSString *dolphinLogPath = @(argv[19]);
    int64_t requestedStartFrame = strtoll(argv[20], NULL, 10);
    int64_t requestedInclusiveEndFrame = strtoll(argv[21], NULL, 10);
    if (processID <= 0 || startupTimeout <= 0.0 || !isfinite(gameSeconds) ||
        gameSeconds <= 0.0 || !isfinite(tailSeconds) || tailSeconds < 0.0 ||
        !isfinite(resumeDelay) || resumeDelay < 0.0 ||
        minimumPixelWidth < 2 || minimumPixelHeight < 2 ||
        requestedCaptureFPS < 60 || requestedCaptureFPS > 240 ||
        !isfinite(playbackEmulationSpeed) || playbackEmulationSpeed <= 0.0 ||
        playbackEmulationSpeed > 1.0 || !isfinite(audioPresentationDelay) ||
        audioPresentationDelay < 0.0 || audioPresentationDelay >= gameSeconds ||
        ptyMasterFD < 0 ||
        fcntl(ptyMasterFD, F_GETFD) < 0 || activePlaybackPath.length == 0 ||
        capturePlaybackPath.length == 0 || dolphinLogPath.length == 0 ||
        requestedStartFrame < INT32_MIN || requestedStartFrame > INT32_MAX ||
        requestedInclusiveEndFrame < requestedStartFrame ||
        requestedInclusiveEndFrame >= INT32_MAX ||
        [activePlaybackPath.stringByStandardizingPath isEqualToString:
            capturePlaybackPath.stringByStandardizingPath] ||
        ((expectedPixelWidth == 0) != (expectedPixelHeight == 0)))
      fail(@"invalid replay recorder argument");
    int64_t commandExclusiveEndFrame = requestedInclusiveEndFrame + 1;
    double frameDerivedGameSeconds =
        (double)(requestedInclusiveEndFrame - requestedStartFrame + 1) / 60.0;
    if (fabs(gameSeconds - frameDerivedGameSeconds) > 1.0 / 60000.0)
      fail(@"declared replay duration does not match its inclusive frame interval");
    int ptyFlags = fcntl(ptyMasterFD, F_GETFL);
    if (ptyFlags < 0 || fcntl(ptyMasterFD, F_SETFL, ptyFlags & ~O_NONBLOCK) != 0)
      fail([NSString stringWithFormat:@"could not make inherited PTY blocking: %s",
                                      strerror(errno)]);

    PlaybackLogMonitor *playbackMonitor =
        [[PlaybackLogMonitor alloc] initWithPTYMasterFD:ptyMasterFD
                                               logPath:dolphinLogPath
                                             processID:processID];
    [playbackMonitor start];
    NSError *playbackFailure = [playbackMonitor failureSnapshot];
    if (playbackFailure)
      fail([NSString stringWithFormat:@"could not start Dolphin PTY monitor: %@",
                                      playbackFailure]);

    activatePlaybackApplication(processID, startupTimeout);
    SCWindow *target = waitForLargestWindow(processID, startupTimeout,
                                            minimumPixelWidth, minimumPixelHeight,
                                            expectedPixelWidth, expectedPixelHeight);
    if (!target)
      fail([NSString stringWithFormat:@"no Slippi Dolphin window appeared for pid %d", processID]);
    fprintf(stderr, "selected window=%u title=%s frame=%.0fx%.0f\n", target.windowID,
            (target.title ?: @"").UTF8String, target.frame.size.width, target.frame.size.height);
    if (kill(processID, SIGSTOP) != 0)
      fail([NSString stringWithFormat:@"could not suspend Slippi Dolphin pid %d", processID]);
    stoppedProcess = processID;

    SCContentFilter *filter = [[SCContentFilter alloc] initWithDesktopIndependentWindow:target];
    SCShareableContentInfo *info = [SCShareableContent infoForFilter:filter];
    double scale = MAX(1.0, info.pointPixelScale);
    size_t width = (size_t)llround(target.frame.size.width * scale);
    size_t height = (size_t)llround(target.frame.size.height * scale);
    width -= width % 2;
    height -= height % 2;
    if (width < minimumPixelWidth || height < minimumPixelHeight)
      fail([NSString stringWithFormat:
          @"Slippi Dolphin playback window is below the required capture size: actual=%zux%zu minimum=%zux%zu",
          width, height, minimumPixelWidth, minimumPixelHeight]);
    if (expectedPixelWidth > 0 &&
        (width != expectedPixelWidth || height != expectedPixelHeight))
      fail([NSString stringWithFormat:
          @"Slippi Dolphin playback window has the wrong capture size: actual=%zux%zu expected=%zux%zu",
          width, height, expectedPixelWidth, expectedPixelHeight]);

    SCStreamConfiguration *streamConfiguration = [SCStreamConfiguration new];
    streamConfiguration.width = width;
    streamConfiguration.height = height;
    // ScreenCaptureKit treats minimumFrameInterval as a delivery throttle;
    // cadence remains best-effort. A 60 Hz threshold can coalesce adjacent 59.94/60 Hz
    // Dolphin presents when compositor jitter places them just inside the
    // threshold.  Requesting 120 Hz leaves margin while Dolphin itself remains
    // the source-of-truth for which frames actually change.
    streamConfiguration.minimumFrameInterval = CMTimeMake(1, requestedCaptureFPS);
    streamConfiguration.queueDepth = 8;
    streamConfiguration.pixelFormat = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
    streamConfiguration.scalesToFit = YES;
    streamConfiguration.preservesAspectRatio = YES;
    streamConfiguration.showsCursor = NO;
    streamConfiguration.capturesAudio = NO;
    streamConfiguration.captureMicrophone = NO;
    streamConfiguration.ignoreShadowsSingleWindow = YES;
    streamConfiguration.shouldBeOpaque = YES;
    streamConfiguration.captureResolution = SCCaptureResolutionBest;

    CaptureWriter *captureWriter =
        [[CaptureWriter alloc] initWithURL:outputURL width:width height:height
                      requestedCaptureFPS:requestedCaptureFPS];
    NSError *writerSetupError = [captureWriter failureSnapshot];
    if (writerSetupError)
      fail([NSString stringWithFormat:@"could not prepare isolated-window writer: %@",
                                      writerSetupError]);
    SCStream *stream = [[SCStream alloc] initWithFilter:filter
                                          configuration:streamConfiguration
                                               delegate:captureWriter];
    NSError *addError = nil;
    if (![stream addStreamOutput:captureWriter type:SCStreamOutputTypeScreen
               sampleHandlerQueue:captureWriter.sampleQueue error:&addError])
      fail([NSString stringWithFormat:@"could not add isolated-window stream output: %@",
                                      addError]);

    dispatch_semaphore_t captureStarted = dispatch_semaphore_create(0);
    __block NSError *startError = nil;
    [stream startCaptureWithCompletionHandler:^(NSError *error) {
      startError = error;
      dispatch_semaphore_signal(captureStarted);
    }];
    dispatch_semaphore_wait(captureStarted, DISPATCH_TIME_FOREVER);
    if (startError)
      fail([NSString stringWithFormat:@"could not start isolated-window capture: %@", startError]);
    int64_t startupNanoseconds = (int64_t)llround(startupTimeout * NSEC_PER_SEC);
    if (dispatch_semaphore_wait(captureWriter.firstFrameReady,
                                dispatch_time(DISPATCH_TIME_NOW, startupNanoseconds)) != 0)
      fail(@"isolated-window stream produced no writable first frame before timeout");
    NSError *captureFailure = [captureWriter failureSnapshot];
    if (captureFailure)
      fail([NSString stringWithFormat:@"recording failed before replay start: %@",
                                      captureFailure]);
    // Dolphin launched against an idle normal-playback command whose replay is
    // empty. ScreenCaptureKit is live before the real command is installed, so
    // the opening emulated frame cannot pass before the capture boundary.
    [NSThread sleepForTimeInterval:resumeDelay];
    if (![playbackMonitor waitUntilQuietForSeconds:PlaybackLogQuietSeconds
                                           timeout:startupTimeout])
      fail(@"Dolphin PTY did not become quiet while the idle command was suspended");
    playbackFailure = [playbackMonitor failureSnapshot];
    if (playbackFailure)
      fail([NSString stringWithFormat:@"Dolphin PTY failed before command install: %@",
                                      playbackFailure]);
    NSDictionary *initialDSPState = zeroAudioBoundaryState(dspAudioPath);
    NSDictionary *initialDTKState = zeroAudioBoundaryState(dtkAudioPath);
    if (![initialDSPState[@"zero_audio_passed"] boolValue] ||
        ![initialDTKState[@"zero_audio_passed"] boolValue])
      fail(@"idle Dolphin command produced visible audio before capture setup");

    NSDictionary *commandInstall = installCapturePlaybackCommand(
        activePlaybackPath, capturePlaybackPath, requestedStartFrame,
        requestedInclusiveEndFrame);
    NSString *expectedReplayPath = commandInstall[@"replay_path"];
    [playbackMonitor armGenerationForReplayPath:expectedReplayPath
                                     startFrame:requestedStartFrame
                              inclusiveEndFrame:requestedInclusiveEndFrame
                            exclusiveEndFrame:commandExclusiveEndFrame];
    playbackFailure = [playbackMonitor failureSnapshot];
    if (playbackFailure)
      fail([NSString stringWithFormat:@"could not arm Dolphin capture generation: %@",
                                      playbackFailure]);
    [playbackMonitor armStopAtFrame:requestedStartFrame];
    if (kill(processID, SIGCONT) != 0)
      fail([NSString stringWithFormat:
          @"could not resume Slippi Dolphin for first synchronized stop pid %d",
          processID]);
    stoppedProcess = 0;
    if (![playbackMonitor waitForArmedStopWithTimeout:startupTimeout]) {
      kill(processID, SIGSTOP);
      stoppedProcess = processID;
      playbackFailure = [playbackMonitor failureSnapshot];
      fail([NSString stringWithFormat:
          @"Dolphin did not stop exactly at requested frame %lld: %@",
          requestedStartFrame, playbackFailure ?: @"timeout"]);
    }
    stoppedProcess = processID;
    if (![playbackMonitor waitUntilQuietForSeconds:PlaybackLogQuietSeconds
                                           timeout:startupTimeout])
      fail(@"Dolphin PTY did not drain cleanly after the first exact stop");
    playbackFailure = [playbackMonitor failureSnapshot];
    if (playbackFailure)
      fail([NSString stringWithFormat:@"first synchronized stop failed: %@",
                                      playbackFailure]);
    NSDictionary *firstStopDSPState = zeroAudioBoundaryState(dspAudioPath);
    NSDictionary *firstStopDTKState = zeroAudioBoundaryState(dtkAudioPath);
    if (![firstStopDSPState[@"zero_audio_passed"] boolValue] ||
        ![firstStopDTKState[@"zero_audio_passed"] boolValue])
      fail(@"Dolphin exposed audio before the requested opening frame");
    NSDictionary *firstStopBoundary = [captureWriter boundarySnapshot];

    // CURRENT_FRAME is emitted before Dolphin applies that frame. Advancing to
    // start+1 and stopping there proves that the requested opening frame was
    // presented while WaveFile still rejected all pre-boundary audio.
    [playbackMonitor armStopAtFrame:requestedStartFrame + 1];
    NSDictionary *beforeAdvanceBoundary = [captureWriter boundarySnapshot];
    if (kill(processID, SIGCONT) != 0)
      fail([NSString stringWithFormat:
          @"could not resume Slippi Dolphin for second synchronized stop pid %d",
          processID]);
    stoppedProcess = 0;
    if (![playbackMonitor waitForArmedStopWithTimeout:startupTimeout]) {
      kill(processID, SIGSTOP);
      stoppedProcess = processID;
      playbackFailure = [playbackMonitor failureSnapshot];
      fail([NSString stringWithFormat:
          @"Dolphin did not stop exactly at opening-frame sentinel %lld: %@",
          requestedStartFrame + 1, playbackFailure ?: @"timeout"]);
    }
    stoppedProcess = processID;
    if (![playbackMonitor waitUntilQuietForSeconds:PlaybackLogQuietSeconds
                                           timeout:startupTimeout])
      fail(@"Dolphin PTY did not drain cleanly after the second exact stop");
    playbackFailure = [playbackMonitor failureSnapshot];
    if (playbackFailure)
      fail([NSString stringWithFormat:@"second synchronized stop failed: %@",
                                      playbackFailure]);
    NSDictionary *secondStopImmediateBoundary = [captureWriter boundarySnapshot];
    NSUInteger secondStopCallbackSequence =
        [secondStopImmediateBoundary[@"callback_count"] unsignedIntegerValue];
    NSUInteger beforeAdvanceCompleteCount =
        [beforeAdvanceBoundary[@"complete_sample_count"] unsignedIntegerValue];
    NSString *beforeSignature = beforeAdvanceBoundary[@"last_visual_signature"];
    NSDictionary *secondStopSettledBoundary = secondStopImmediateBoundary;
    BOOL freshCallbackAfterSecondStop = NO;
    BOOL completeAdvanced = NO;
    BOOL displayAdvanced = NO;
    BOOL signatureChanged = NO;
    BOOL visualSignatureConcrete = NO;
    NSDate *openingFrameDeadline =
        [NSDate dateWithTimeIntervalSinceNow:startupTimeout];
    while ([openingFrameDeadline timeIntervalSinceNow] > 0.0) {
      captureFailure = [captureWriter failureSnapshot];
      if (captureFailure)
        fail([NSString stringWithFormat:
            @"recording failed while establishing the opening-frame boundary: %@",
            captureFailure]);
      secondStopSettledBoundary = [captureWriter boundarySnapshot];
      freshCallbackAfterSecondStop =
          [secondStopSettledBoundary[@"callback_count"] unsignedIntegerValue] >
          secondStopCallbackSequence;
      completeAdvanced =
          [secondStopSettledBoundary[@"complete_sample_count"] unsignedIntegerValue] >
          beforeAdvanceCompleteCount;
      displayAdvanced =
          [secondStopSettledBoundary[@"last_display_time_mach"] unsignedLongLongValue] >
          [beforeAdvanceBoundary[@"last_display_time_mach"] unsignedLongLongValue];
      NSString *afterSignature = secondStopSettledBoundary[@"last_visual_signature"];
      signatureChanged = ![beforeSignature isEqualToString:afterSignature] &&
          ![beforeSignature isEqualToString:@"0000000000000000"] &&
          ![afterSignature isEqualToString:@"0000000000000000"];
      NSString *afterContentClass =
          secondStopSettledBoundary[@"last_content_class"];
      visualSignatureConcrete =
          afterSignature.length == 16 &&
          ![afterSignature isEqualToString:@"0000000000000000"] &&
          [@[@"A", @"W", @"K", @"N"] containsObject:afterContentClass];
      if (freshCallbackAfterSecondStop && completeAdvanced && displayAdvanced &&
          visualSignatureConcrete)
        break;
      [NSThread sleepForTimeInterval:0.002];
    }
    if (!freshCallbackAfterSecondStop || !completeAdvanced || !displayAdvanced ||
        !visualSignatureConcrete)
      fail([NSString stringWithFormat:
          @"opening-frame presentation proof failed: fresh=%d complete=%d display=%d concrete=%d signature_changed=%d",
          freshCallbackAfterSecondStop, completeAdvanced, displayAdvanced,
          visualSignatureConcrete, signatureChanged]);
    NSDictionary *secondStopDSPState = zeroAudioBoundaryState(dspAudioPath);
    NSDictionary *secondStopDTKState = zeroAudioBoundaryState(dtkAudioPath);
    if (![secondStopDSPState[@"zero_audio_passed"] boolValue] ||
        ![secondStopDTKState[@"zero_audio_passed"] boolValue])
      fail(@"Dolphin exposed audio before the logical zero-audio barrier");

    const int64_t dspBarrierFrames = 0;
    const int64_t dtkBarrierFrames = 0;
    CMTime barrierHostTime = CMClockGetTime(CMClockGetHostTimeClock());
    NSDictionary *armedOpeningBoundary = [captureWriter armGameplayAtHostTime:barrierHostTime];
    if (!armedOpeningBoundary) fail(@"could not establish immutable startup phase boundary");
    double videoAlignmentOffset = [armedOpeningBoundary[@"last_relative_display_seconds"] doubleValue];
    if (!isfinite(videoAlignmentOffset)) fail(@"capture has no concrete host-time origin");
    NSMutableArray<NSDictionary *> *clockLandmarks = [NSMutableArray array];
    appendClockLandmark(clockLandmarks, videoAlignmentOffset, 0.0);
    [playbackMonitor disarmStopTarget];
    // Dolphin's command endFrame remains an exclusive control boundary. Its
    // natural CURRENT_FRAME trace ends at the requested inclusive content
    // frame, so the terminal stop and trace proof target that generated frame.
    [playbackMonitor armStopAtFrame:requestedInclusiveEndFrame];
    if (kill(processID, SIGCONT) != 0)
      fail([NSString stringWithFormat:@"could not resume Slippi Dolphin after audio barrier pid %d",
                                      processID]);
    stoppedProcess = 0;

    const double completionTolerance = 0.5;
    const double landmarkInterval = 5.0;
    // Leave enough scheduling margin for the encoded timeline itself to retain
    // the requested tail after the final audio-clock landmark.
    const double requiredStableSeconds = MAX(tailSeconds + 0.100, 0.350);
    const double sharedBarrierSeconds = 0.0;
    WaveLayout dspLayout = {NO, 0, 0, 0};
    WaveLayout dtkLayout = {NO, 0, 0, 0};
    BOOL audioClockInitialized = NO;
    double firstMalformedDSPAt = NAN;
    double firstMalformedDTKAt = NAN;
    double nextLandmarkAudioSeconds = landmarkInterval;
    double lastMixedAudioSeconds = 0.0;
    double lastDspSeconds = 0.0;
    double lastDtkSeconds = 0.0;
    double lastAnyAudioGrowthAt = CFAbsoluteTimeGetCurrent();
    double lastMixedAudioGrowthSourceSeconds = videoAlignmentOffset;
    BOOL audioEndObserved = NO;
    BOOL terminalStopObserved = NO;
    NSDictionary *terminalStopBoundary = nil;
    NSDictionary *terminalBarrierBoundary = nil;
    NSDictionary *terminalAppliedBoundary = nil;
    NSDictionary *terminalStoppedHoldBoundary = nil;
    NSDictionary *terminalFinalFrozenBoundary = nil;
    NSDictionary *terminalFinalFrozenTailProof = nil;
    NSDictionary *terminalSealedStopSuffixProof = nil;
    BOOL terminalFreshCallbackAfterResume = NO;
    BOOL terminalPreResumeFrozenCallbackAdvanced = NO;
    BOOL terminalPreResumeFrozenOnlyRetainedCallbacks = NO;
    BOOL terminalPreResumeFrozenTransitionCountersUnchanged = NO;
    BOOL terminalPreResumeFrozenVisualStateRetained = NO;
    NSUInteger terminalPreResumeFrozenCompleteDelta = 0;
    NSUInteger terminalPreResumeFrozenIdleDelta = 0;
    BOOL terminalCompleteAdvanced = NO;
    BOOL terminalCompleteCallbackSequenceAdvanced = NO;
    BOOL terminalCompleteDisplayAfterResume = NO;
    BOOL terminalStoppedHoldCallbackAdvanced = NO;
    BOOL terminalStoppedHoldOnlyRetainedCallbacks = NO;
    BOOL terminalStoppedHoldTransitionCountersUnchanged = NO;
    BOOL terminalStoppedHoldCompleteIdentityConsistent = NO;
    BOOL terminalStoppedHoldDisplayIntervalCovered = NO;
    BOOL terminalStoppedHoldVisualStateRetained = NO;
    NSUInteger terminalStoppedHoldCallbackDelta = 0;
    NSUInteger terminalStoppedHoldPostAcceptCallbackDelta = 0;
    NSUInteger terminalStoppedHoldCompleteDelta = 0;
    NSUInteger terminalStoppedHoldIdleDelta = 0;
    BOOL terminalParentStopSignalSucceeded = NO;
    uint64_t terminalResumeHostTimeMach = 0;
    double terminalResumeHostTimeSeconds = NAN;
    double terminalPostResumeWallSeconds = 0.0;
    double terminalStoppedDisplayHoldSeconds = 0.0;
    const double terminalRequiredStoppedDisplayHoldSeconds =
        1.0 / (60.0 * playbackEmulationSpeed);
    const double terminalExpectedRawAudioSeconds =
        gameSeconds - audioPresentationDelay;
    const double terminalVisibleAudioLowerBoundSeconds =
        terminalExpectedRawAudioSeconds -
        TerminalAudioFinalizedBufferAllowanceSeconds;
    const double terminalVisibleAudioUpperBoundSeconds =
        terminalExpectedRawAudioSeconds +
        TerminalAudioMaximumPhysicalOvershootSeconds;
    BOOL terminalVisibleAudioWithinBoundsBeforeParentStop = NO;
    BOOL terminalAudioPostStopNoGrowth = NO;
    double terminalDSPEndpointSeconds = NAN;
    double terminalDTKEndpointSeconds = NAN;
    double terminalDSPPhysicalOffsetSeconds = NAN;
    double terminalDTKPhysicalOffsetSeconds = NAN;
    int64_t terminalDSPFramesBeforeParentStop = -1;
    int64_t terminalDTKFramesBeforeParentStop = -1;
    int64_t terminalDSPFramesAtStopBaseline = -1;
    int64_t terminalDTKFramesAtStopBaseline = -1;
    int64_t terminalDSPFramesAfterObservation = -1;
    int64_t terminalDTKFramesAfterObservation = -1;
    double terminalStopObservedAt = NAN;
    double terminalStopSourceVideoSeconds = NAN;
    double terminalCaptureTailSeconds = 0.0;
    double terminalCaptureTailWallSeconds = 0.0;
    // Video rendering intentionally runs Dolphin at half speed so each
    // emulated frame remains visible across at least two compositor refreshes.
    // The mux later maps wall-clock video back to Dolphin's audio clock.
    double maximumPlaybackSeconds =
        gameSeconds / playbackEmulationSpeed + startupTimeout;
    NSDate *playbackDeadline =
        [NSDate dateWithTimeIntervalSinceNow:maximumPlaybackSeconds + requiredStableSeconds];
    while ([playbackDeadline timeIntervalSinceNow] > 0.0) {
      if (!terminalStopObserved && [playbackMonitor stopTargetReachedSnapshot]) {
        stoppedProcess = processID;
        terminalStopBoundary = [captureWriter boundarySnapshot];
        NSUInteger terminalStopCallbackSequence =
            [terminalStopBoundary[@"callback_count"] unsignedIntegerValue];
        if (![captureWriter waitForCallbackAfterSequence:terminalStopCallbackSequence
                                                  timeout:MIN(
                                                      startupTimeout,
                                                      TerminalApplicationProofMaximumSeconds)]) {
          captureFailure = [captureWriter failureSnapshot];
          fail([NSString stringWithFormat:
              @"terminal pre-resume frozen callback proof failed: %@",
              captureFailure ?: captureError(@"no fresh ScreenCaptureKit callback while Dolphin was stopped")]);
        }
        terminalBarrierBoundary =
            [captureWriter boundarySnapshotAndArmTerminalGapRecovery];
        NSUInteger terminalStopCompleteCount =
            [terminalStopBoundary[@"complete_sample_count"] unsignedIntegerValue];
        NSUInteger terminalBarrierCompleteCountForFreeze =
            [terminalBarrierBoundary[@"complete_sample_count"] unsignedIntegerValue];
        NSUInteger terminalStopIdleCount =
            [terminalStopBoundary[@"idle_sample_count"] unsignedIntegerValue];
        NSUInteger terminalBarrierIdleCountForFreeze =
            [terminalBarrierBoundary[@"idle_sample_count"] unsignedIntegerValue];
        terminalPreResumeFrozenCallbackAdvanced =
            [terminalBarrierBoundary[@"callback_count"] unsignedIntegerValue] >
            terminalStopCallbackSequence;
        if (terminalBarrierCompleteCountForFreeze >= terminalStopCompleteCount &&
            terminalBarrierIdleCountForFreeze >= terminalStopIdleCount) {
          terminalPreResumeFrozenCompleteDelta =
              terminalBarrierCompleteCountForFreeze - terminalStopCompleteCount;
          terminalPreResumeFrozenIdleDelta =
              terminalBarrierIdleCountForFreeze - terminalStopIdleCount;
          terminalPreResumeFrozenOnlyRetainedCallbacks =
              [terminalBarrierBoundary[@"callback_count"] unsignedIntegerValue] -
                  terminalStopCallbackSequence ==
              terminalPreResumeFrozenCompleteDelta +
                  terminalPreResumeFrozenIdleDelta;
        }
        terminalPreResumeFrozenTransitionCountersUnchanged =
            [terminalBarrierBoundary[@"visual_signature_transition_count"]
                unsignedIntegerValue] ==
                [terminalStopBoundary[@"visual_signature_transition_count"]
                    unsignedIntegerValue] &&
            [terminalBarrierBoundary[@"content_class_transition_count"]
                unsignedIntegerValue] ==
                [terminalStopBoundary[@"content_class_transition_count"]
                    unsignedIntegerValue];
        terminalPreResumeFrozenVisualStateRetained =
            [terminalBarrierBoundary[@"last_visual_signature"]
                isEqualToString:terminalStopBoundary[@"last_visual_signature"]] &&
            [terminalBarrierBoundary[@"last_content_class"]
                isEqualToString:terminalStopBoundary[@"last_content_class"]];
        if (!terminalPreResumeFrozenCallbackAdvanced ||
            !terminalPreResumeFrozenOnlyRetainedCallbacks ||
            !terminalPreResumeFrozenTransitionCountersUnchanged ||
            !terminalPreResumeFrozenVisualStateRetained)
          fail([NSString stringWithFormat:
              @"terminal pre-resume frozen callback was not retained: advanced=%d retained=%d transitions=%d visual=%d complete_delta=%zu idle_delta=%zu",
              terminalPreResumeFrozenCallbackAdvanced,
              terminalPreResumeFrozenOnlyRetainedCallbacks,
              terminalPreResumeFrozenTransitionCountersUnchanged,
              terminalPreResumeFrozenVisualStateRetained,
              terminalPreResumeFrozenCompleteDelta,
              terminalPreResumeFrozenIdleDelta]);
        NSUInteger terminalBarrierCallbackSequence =
            [terminalBarrierBoundary[@"callback_count"] unsignedIntegerValue];
        NSUInteger terminalBarrierCompleteCount =
            [terminalBarrierBoundary[@"complete_sample_count"] unsignedIntegerValue];
        [playbackMonitor disarmStopTarget];
        CMTime terminalResumeHostTime =
            CMClockGetTime(CMClockGetHostTimeClock());
        terminalResumeHostTimeMach =
            CMClockConvertHostTimeToSystemUnits(terminalResumeHostTime);
        terminalResumeHostTimeSeconds =
            CMTimeGetSeconds(terminalResumeHostTime);
        if (kill(processID, SIGCONT) != 0)
          fail([NSString stringWithFormat:
              @"could not resume Slippi Dolphin after inclusive terminal frame barrier pid %d",
              processID]);
        stoppedProcess = 0;
        CFAbsoluteTime terminalResumeWallTime = CFAbsoluteTimeGetCurrent();
        NSDate *terminalApplyDeadline =
            [NSDate dateWithTimeIntervalSinceNow:
                MIN(startupTimeout, TerminalApplicationProofMaximumSeconds)];
        terminalAppliedBoundary = terminalBarrierBoundary;
        while ([terminalApplyDeadline timeIntervalSinceNow] > 0.0) {
          captureFailure = [captureWriter failureSnapshot];
          if (captureFailure) {
            kill(processID, SIGSTOP);
            stoppedProcess = processID;
            fail([NSString stringWithFormat:
                @"recording failed while proving the inclusive terminal frame was applied: %@",
                captureFailure]);
          }
          playbackFailure = [playbackMonitor failureSnapshot];
          if (playbackFailure) {
            kill(processID, SIGSTOP);
            stoppedProcess = processID;
            fail([NSString stringWithFormat:
                @"Dolphin trace failed while proving the inclusive terminal frame was applied: %@",
                playbackFailure]);
          }
          terminalAppliedBoundary = [captureWriter boundarySnapshot];
          terminalFreshCallbackAfterResume =
              [terminalAppliedBoundary[@"callback_count"] unsignedIntegerValue] >
              terminalBarrierCallbackSequence;
          terminalCompleteAdvanced =
              [terminalAppliedBoundary[@"complete_sample_count"] unsignedIntegerValue] >
              terminalBarrierCompleteCount;
          terminalCompleteCallbackSequenceAdvanced =
              [terminalAppliedBoundary[@"last_complete_callback_sequence"]
                  unsignedIntegerValue] > terminalBarrierCallbackSequence;
          terminalCompleteDisplayAfterResume =
              [terminalAppliedBoundary[@"last_complete_display_time_mach"]
                  unsignedLongLongValue] > terminalResumeHostTimeMach;
          terminalPostResumeWallSeconds =
              MAX(0.0, CFAbsoluteTimeGetCurrent() - terminalResumeWallTime);
          if (!dspLayout.valid) dspLayout = waveLayoutAtPath(dspAudioPath);
          if (!dtkLayout.valid) dtkLayout = waveLayoutAtPath(dtkAudioPath);
          if (dspLayout.valid && dtkLayout.valid) {
            int64_t terminalDSPFrames = waveFramesAtPath(dspAudioPath, dspLayout);
            int64_t terminalDTKFrames = waveFramesAtPath(dtkAudioPath, dtkLayout);
            if (terminalDSPFrames >= 0 && terminalDTKFrames >= 0) {
              terminalDSPFramesBeforeParentStop = terminalDSPFrames;
              terminalDTKFramesBeforeParentStop = terminalDTKFrames;
              terminalDSPEndpointSeconds =
                  (double)terminalDSPFrames / dspLayout.sampleRate;
              terminalDTKEndpointSeconds =
                  (double)terminalDTKFrames / dtkLayout.sampleRate;
              terminalDSPPhysicalOffsetSeconds =
                  terminalDSPEndpointSeconds - terminalExpectedRawAudioSeconds;
              terminalDTKPhysicalOffsetSeconds =
                  terminalDTKEndpointSeconds - terminalExpectedRawAudioSeconds;
              if (terminalDSPEndpointSeconds >
                      terminalVisibleAudioUpperBoundSeconds ||
                  terminalDTKEndpointSeconds >
                      terminalVisibleAudioUpperBoundSeconds) {
                kill(processID, SIGSTOP);
                stoppedProcess = processID;
                fail([NSString stringWithFormat:
                    @"visible terminal audio exceeded its immediate upper guard: DSP=%.6f DTK=%.6f upper=%.6f",
                    terminalDSPEndpointSeconds, terminalDTKEndpointSeconds,
                    terminalVisibleAudioUpperBoundSeconds]);
              }
              terminalVisibleAudioWithinBoundsBeforeParentStop =
                  terminalDSPEndpointSeconds >=
                      terminalVisibleAudioLowerBoundSeconds &&
                  terminalDTKEndpointSeconds >=
                      terminalVisibleAudioLowerBoundSeconds;
            }
          }
          if (terminalFreshCallbackAfterResume && terminalCompleteAdvanced &&
              terminalCompleteCallbackSequenceAdvanced &&
              terminalCompleteDisplayAfterResume &&
              terminalVisibleAudioWithinBoundsBeforeParentStop)
            break;
          [NSThread sleepForTimeInterval:0.002];
        }
        if (!terminalFreshCallbackAfterResume || !terminalCompleteAdvanced ||
            !terminalCompleteCallbackSequenceAdvanced ||
            !terminalCompleteDisplayAfterResume ||
            !terminalVisibleAudioWithinBoundsBeforeParentStop) {
          kill(processID, SIGSTOP);
          stoppedProcess = processID;
          fail([NSString stringWithFormat:
              @"inclusive terminal frame application proof failed: fresh=%d complete=%d complete_sequence=%d complete_display=%d resume_to_complete_wall=%.6f audio_bounds=%d DSP=%.6f DTK=%.6f lower=%.6f upper=%.6f",
              terminalFreshCallbackAfterResume, terminalCompleteAdvanced,
              terminalCompleteCallbackSequenceAdvanced,
              terminalCompleteDisplayAfterResume,
              terminalPostResumeWallSeconds,
              terminalVisibleAudioWithinBoundsBeforeParentStop,
              terminalDSPEndpointSeconds, terminalDTKEndpointSeconds,
              terminalVisibleAudioLowerBoundSeconds,
              terminalVisibleAudioUpperBoundSeconds]);
        }
        int terminalParentStopResult = kill(processID, SIGSTOP);
        terminalParentStopSignalSucceeded = terminalParentStopResult == 0;
        if (!terminalParentStopSignalSucceeded)
          fail([NSString stringWithFormat:
              @"could not suspend Slippi Dolphin after inclusive terminal frame application: %s",
              strerror(errno)]);
        stoppedProcess = processID;
        [NSThread sleepForTimeInterval:
            TerminalAudioStopStabilityIntervalSeconds];
        terminalDSPFramesAtStopBaseline =
            waveFramesAtPath(dspAudioPath, dspLayout);
        terminalDTKFramesAtStopBaseline =
            waveFramesAtPath(dtkAudioPath, dtkLayout);
        double terminalDSPStopBaselineSeconds =
            terminalDSPFramesAtStopBaseline >= 0 ?
            (double)terminalDSPFramesAtStopBaseline / dspLayout.sampleRate : NAN;
        double terminalDTKStopBaselineSeconds =
            terminalDTKFramesAtStopBaseline >= 0 ?
            (double)terminalDTKFramesAtStopBaseline / dtkLayout.sampleRate : NAN;
        if (!isfinite(terminalDSPStopBaselineSeconds) ||
            !isfinite(terminalDTKStopBaselineSeconds) ||
            terminalDSPFramesAtStopBaseline <
                terminalDSPFramesBeforeParentStop ||
            terminalDTKFramesAtStopBaseline <
                terminalDTKFramesBeforeParentStop ||
            terminalDSPStopBaselineSeconds <
                terminalVisibleAudioLowerBoundSeconds ||
            terminalDTKStopBaselineSeconds <
                terminalVisibleAudioLowerBoundSeconds ||
            terminalDSPStopBaselineSeconds >
                terminalVisibleAudioUpperBoundSeconds ||
            terminalDTKStopBaselineSeconds >
                terminalVisibleAudioUpperBoundSeconds)
          fail([NSString stringWithFormat:
              @"visible terminal audio left its bounds or moved backward at the stopped baseline: before_DSP=%lld before_DTK=%lld baseline_DSP=%lld baseline_DTK=%lld DSP=%.6f DTK=%.6f lower=%.6f upper=%.6f",
              terminalDSPFramesBeforeParentStop,
              terminalDTKFramesBeforeParentStop,
              terminalDSPFramesAtStopBaseline,
              terminalDTKFramesAtStopBaseline,
              terminalDSPStopBaselineSeconds,
              terminalDTKStopBaselineSeconds,
              terminalVisibleAudioLowerBoundSeconds,
              terminalVisibleAudioUpperBoundSeconds]);
        [NSThread sleepForTimeInterval:
            TerminalAudioStopStabilityIntervalSeconds];
        terminalDSPFramesAfterObservation =
            waveFramesAtPath(dspAudioPath, dspLayout);
        terminalDTKFramesAfterObservation =
            waveFramesAtPath(dtkAudioPath, dtkLayout);
        terminalStoppedHoldBoundary = [captureWriter boundarySnapshot];
        captureFailure = [captureWriter failureSnapshot];
        if (captureFailure)
          fail([NSString stringWithFormat:
              @"recording failed during the terminal stopped hold: %@",
              captureFailure]);
        NSUInteger terminalAppliedCallbackCount =
            [terminalAppliedBoundary[@"callback_count"] unsignedIntegerValue];
        NSUInteger terminalStoppedHoldCallbackCount =
            [terminalStoppedHoldBoundary[@"callback_count"] unsignedIntegerValue];
        NSUInteger terminalAppliedCompleteCount =
            [terminalAppliedBoundary[@"complete_sample_count"] unsignedIntegerValue];
        NSUInteger terminalStoppedHoldCompleteCount =
            [terminalStoppedHoldBoundary[@"complete_sample_count"] unsignedIntegerValue];
        NSUInteger terminalAppliedIdleCount =
            [terminalAppliedBoundary[@"idle_sample_count"] unsignedIntegerValue];
        NSUInteger terminalStoppedHoldIdleCount =
            [terminalStoppedHoldBoundary[@"idle_sample_count"] unsignedIntegerValue];
        terminalStoppedHoldCallbackAdvanced =
            terminalStoppedHoldCallbackCount > terminalAppliedCallbackCount;
        if (terminalStoppedHoldCallbackCount >= terminalAppliedCallbackCount &&
            terminalStoppedHoldCompleteCount >= terminalAppliedCompleteCount &&
            terminalStoppedHoldIdleCount >= terminalAppliedIdleCount) {
          terminalStoppedHoldCallbackDelta =
              terminalStoppedHoldCallbackCount - terminalAppliedCallbackCount;
          terminalStoppedHoldCompleteDelta =
              terminalStoppedHoldCompleteCount - terminalAppliedCompleteCount;
          terminalStoppedHoldIdleDelta =
              terminalStoppedHoldIdleCount - terminalAppliedIdleCount;
          terminalStoppedHoldOnlyRetainedCallbacks =
              terminalStoppedHoldCallbackDelta ==
                  terminalStoppedHoldCompleteDelta + terminalStoppedHoldIdleDelta;
        }
        terminalStoppedHoldTransitionCountersUnchanged =
            [terminalStoppedHoldBoundary[@"visual_signature_transition_count"]
                unsignedIntegerValue] ==
                [terminalAppliedBoundary[@"visual_signature_transition_count"]
                    unsignedIntegerValue] &&
            [terminalStoppedHoldBoundary[@"content_class_transition_count"]
                unsignedIntegerValue] ==
                [terminalAppliedBoundary[@"content_class_transition_count"]
                    unsignedIntegerValue];
        NSUInteger terminalAcceptedCompleteSequence =
            [terminalAppliedBoundary[@"last_complete_callback_sequence"]
                unsignedIntegerValue];
        if (terminalStoppedHoldCallbackCount >= terminalAcceptedCompleteSequence)
          terminalStoppedHoldPostAcceptCallbackDelta =
              terminalStoppedHoldCallbackCount - terminalAcceptedCompleteSequence;
        NSUInteger terminalHoldLastCompleteSequence =
            [terminalStoppedHoldBoundary[@"last_complete_callback_sequence"]
                unsignedIntegerValue];
        uint64_t terminalAppliedCompleteDisplayTime =
            [terminalAppliedBoundary[@"last_complete_display_time_mach"]
                unsignedLongLongValue];
        uint64_t terminalHoldLastCompleteDisplayTime =
            [terminalStoppedHoldBoundary[@"last_complete_display_time_mach"]
                unsignedLongLongValue];
        terminalStoppedHoldCompleteIdentityConsistent =
            (terminalStoppedHoldCompleteDelta == 0 &&
             terminalHoldLastCompleteSequence == terminalAcceptedCompleteSequence &&
             terminalHoldLastCompleteDisplayTime == terminalAppliedCompleteDisplayTime) ||
            (terminalStoppedHoldCompleteDelta > 0 &&
             terminalHoldLastCompleteSequence > terminalAcceptedCompleteSequence &&
             terminalHoldLastCompleteDisplayTime > terminalAppliedCompleteDisplayTime);
        uint64_t terminalAcceptedCompleteDisplayTime =
            [terminalAppliedBoundary[@"last_complete_display_time_mach"]
                unsignedLongLongValue];
        uint64_t terminalStoppedHoldDisplayTime =
            [terminalStoppedHoldBoundary[@"last_display_time_mach"]
                unsignedLongLongValue];
        if (terminalStoppedHoldDisplayTime >
            terminalAcceptedCompleteDisplayTime) {
          terminalStoppedDisplayHoldSeconds = CMTimeGetSeconds(CMTimeSubtract(
              CMClockMakeHostTimeFromSystemUnits(terminalStoppedHoldDisplayTime),
              CMClockMakeHostTimeFromSystemUnits(
                  terminalAcceptedCompleteDisplayTime)));
        } else {
          terminalStoppedDisplayHoldSeconds = 0.0;
        }
        terminalStoppedHoldDisplayIntervalCovered =
            isfinite(terminalStoppedDisplayHoldSeconds) &&
            terminalStoppedDisplayHoldSeconds >=
                terminalRequiredStoppedDisplayHoldSeconds;
        NSString *terminalAcceptedSignature =
            terminalAppliedBoundary[@"last_visual_signature"];
        NSString *terminalStoppedSignature =
            terminalStoppedHoldBoundary[@"last_visual_signature"];
        NSString *terminalAcceptedClass =
            terminalAppliedBoundary[@"last_content_class"];
        NSString *terminalStoppedClass =
            terminalStoppedHoldBoundary[@"last_content_class"];
        terminalStoppedHoldVisualStateRetained =
            terminalAcceptedSignature.length == 16 &&
            ![terminalAcceptedSignature isEqualToString:@"0000000000000000"] &&
            [terminalAcceptedSignature isEqualToString:terminalStoppedSignature] &&
            [@[@"A", @"W", @"K", @"N"]
                containsObject:terminalAcceptedClass] &&
            [terminalAcceptedClass isEqualToString:terminalStoppedClass];
        terminalAudioPostStopNoGrowth =
            terminalDSPFramesAtStopBaseline >= 0 &&
            terminalDTKFramesAtStopBaseline >= 0 &&
            terminalDSPFramesAfterObservation == terminalDSPFramesAtStopBaseline &&
            terminalDTKFramesAfterObservation == terminalDTKFramesAtStopBaseline;
        if (terminalDSPFramesAfterObservation >= 0 &&
            terminalDTKFramesAfterObservation >= 0) {
          terminalDSPEndpointSeconds =
              (double)terminalDSPFramesAfterObservation / dspLayout.sampleRate;
          terminalDTKEndpointSeconds =
              (double)terminalDTKFramesAfterObservation / dtkLayout.sampleRate;
          terminalDSPPhysicalOffsetSeconds =
              terminalDSPEndpointSeconds - terminalExpectedRawAudioSeconds;
          terminalDTKPhysicalOffsetSeconds =
              terminalDTKEndpointSeconds - terminalExpectedRawAudioSeconds;
        }
        if (!terminalStoppedHoldCallbackAdvanced ||
            !terminalStoppedHoldOnlyRetainedCallbacks ||
            !terminalStoppedHoldTransitionCountersUnchanged ||
            !terminalStoppedHoldCompleteIdentityConsistent ||
            !terminalStoppedHoldDisplayIntervalCovered ||
            !terminalStoppedHoldVisualStateRetained ||
            !terminalAudioPostStopNoGrowth ||
            !isfinite(terminalDSPEndpointSeconds) ||
            !isfinite(terminalDTKEndpointSeconds) ||
            terminalDSPEndpointSeconds <
                terminalVisibleAudioLowerBoundSeconds ||
            terminalDTKEndpointSeconds <
                terminalVisibleAudioLowerBoundSeconds ||
            terminalDSPEndpointSeconds >
                terminalVisibleAudioUpperBoundSeconds ||
            terminalDTKEndpointSeconds >
                terminalVisibleAudioUpperBoundSeconds)
          fail([NSString stringWithFormat:
              @"terminal stopped-hold boundary failed: callback=%d retained_callbacks=%d transitions_unchanged=%d complete_identity=%d callback_delta=%zu complete_delta=%zu idle_delta=%zu display_hold=%d visual_retained=%d hold_seconds=%.6f required=%.6f no_audio_growth=%d DSP=%.6f DTK=%.6f lower=%.6f upper=%.6f",
              terminalStoppedHoldCallbackAdvanced,
              terminalStoppedHoldOnlyRetainedCallbacks,
              terminalStoppedHoldTransitionCountersUnchanged,
              terminalStoppedHoldCompleteIdentityConsistent,
              terminalStoppedHoldCallbackDelta,
              terminalStoppedHoldCompleteDelta,
              terminalStoppedHoldIdleDelta,
              terminalStoppedHoldDisplayIntervalCovered,
              terminalStoppedHoldVisualStateRetained,
              terminalStoppedDisplayHoldSeconds,
              terminalRequiredStoppedDisplayHoldSeconds,
              terminalAudioPostStopNoGrowth, terminalDSPEndpointSeconds,
              terminalDTKEndpointSeconds, terminalVisibleAudioLowerBoundSeconds,
              terminalVisibleAudioUpperBoundSeconds]);
        terminalStopObserved = YES;
        terminalStopObservedAt = CFAbsoluteTimeGetCurrent();
        terminalStopSourceVideoSeconds =
            MAX(videoAlignmentOffset, [captureWriter currentSourceVideoSeconds]);
      }
      captureFailure = [captureWriter failureSnapshot];
      if (captureFailure)
        fail([NSString stringWithFormat:@"recording failed during replay playback: %@",
                                        captureFailure]);
      playbackFailure = [playbackMonitor failureSnapshot];
      if (playbackFailure) {
        kill(processID, SIGSTOP);
        stoppedProcess = processID;
        fail([NSString stringWithFormat:@"Dolphin playback trace failed: %@",
                                        playbackFailure]);
      }
      double now = CFAbsoluteTimeGetCurrent();
      if (terminalStopObserved)
        terminalCaptureTailWallSeconds =
            MAX(0.0, now - terminalStopObservedAt);
      if (!dspLayout.valid) dspLayout = waveLayoutAtPath(dspAudioPath);
      if (!dtkLayout.valid) dtkLayout = waveLayoutAtPath(dtkAudioPath);
      struct stat dspStatus;
      struct stat dtkStatus;
      uint64_t dspSize = stat(dspAudioPath.fileSystemRepresentation, &dspStatus) == 0 &&
              dspStatus.st_size >= 0 ? (uint64_t)dspStatus.st_size : 0;
      uint64_t dtkSize = stat(dtkAudioPath.fileSystemRepresentation, &dtkStatus) == 0 &&
              dtkStatus.st_size >= 0 ? (uint64_t)dtkStatus.st_size : 0;
      if (!dspLayout.valid && dspSize > 0 && !isfinite(firstMalformedDSPAt))
        firstMalformedDSPAt = now;
      if (!dtkLayout.valid && dtkSize > 0 && !isfinite(firstMalformedDTKAt))
        firstMalformedDTKAt = now;
      if ((!dspLayout.valid && isfinite(firstMalformedDSPAt) &&
           now - firstMalformedDSPAt > 0.5) ||
          (!dtkLayout.valid && isfinite(firstMalformedDTKAt) &&
           now - firstMalformedDTKAt > 0.5))
        fail(@"Dolphin exposed a persistent malformed growing WAVE header");
      if (!dspLayout.valid || !dtkLayout.valid) {
        [NSThread sleepForTimeInterval:0.020];
        continue;
      }
      int64_t dspFrames = waveFramesAtPath(dspAudioPath, dspLayout);
      int64_t dtkFrames = waveFramesAtPath(dtkAudioPath, dtkLayout);
      if (dspFrames < dspBarrierFrames || dtkFrames < dtkBarrierFrames)
        fail(@"Dolphin audio clock moved before its synchronization barrier");
      if (!audioClockInitialized) {
        audioClockInitialized = YES;
        lastAnyAudioGrowthAt = now;
      }
      double dspSeconds =
          MAX(0.0, (double)dspFrames / dspLayout.sampleRate - sharedBarrierSeconds);
      double dtkSeconds =
          MAX(0.0, (double)dtkFrames / dtkLayout.sampleRate - sharedBarrierSeconds);
      double mixedAudioSeconds = MAX(dspSeconds, dtkSeconds);
      double sourceVideoSeconds =
          MAX(videoAlignmentOffset, [captureWriter currentSourceVideoSeconds]);
      if (terminalStopObserved)
        terminalCaptureTailSeconds =
            MAX(0.0, sourceVideoSeconds - terminalStopSourceVideoSeconds);
      if (dspSeconds > lastDspSeconds + 0.000001 ||
          dtkSeconds > lastDtkSeconds + 0.000001) {
        lastAnyAudioGrowthAt = now;
        lastDspSeconds = MAX(lastDspSeconds, dspSeconds);
        lastDtkSeconds = MAX(lastDtkSeconds, dtkSeconds);
      }
      if (mixedAudioSeconds > lastMixedAudioSeconds + 0.000001) {
        lastMixedAudioSeconds = mixedAudioSeconds;
        lastMixedAudioGrowthSourceSeconds = sourceVideoSeconds;
        if (lastMixedAudioSeconds >= nextLandmarkAudioSeconds) {
          appendClockLandmark(clockLandmarks, sourceVideoSeconds, lastMixedAudioSeconds);
          while (nextLandmarkAudioSeconds <= lastMixedAudioSeconds)
            nextLandmarkAudioSeconds += landmarkInterval;
        }
      }
      double stableSeconds = now - lastAnyAudioGrowthAt;
      if (stableSeconds >= requiredStableSeconds &&
          (dspSeconds + completionTolerance < gameSeconds ||
           dtkSeconds + completionTolerance < gameSeconds))
        fail([NSString stringWithFormat:
            @"Dolphin replay ended before its declared audio duration: expected=%.6f DSP=%.6f DTK=%.6f stable=%.6f",
            gameSeconds, dspSeconds, dtkSeconds, stableSeconds]);
      if (dspSeconds + completionTolerance >= gameSeconds &&
          dtkSeconds + completionTolerance >= gameSeconds &&
          stableSeconds >= requiredStableSeconds && terminalStopObserved &&
          terminalCaptureTailSeconds >= requiredStableSeconds &&
          terminalCaptureTailWallSeconds >= requiredStableSeconds) {
        if (![playbackMonitor waitForGenerationTraceWithTimeout:startupTimeout]) {
          playbackFailure = [playbackMonitor failureSnapshot];
          fail([NSString stringWithFormat:
              @"Dolphin replay frame trace did not reach inclusive last generated content frame %lld; command boundary remains exclusive at %lld: %@",
              requestedInclusiveEndFrame, commandExclusiveEndFrame,
              playbackFailure ?: @"timeout"]);
        }
        if (![playbackMonitor waitUntilQuietForSeconds:PlaybackLogQuietSeconds
                                               timeout:startupTimeout])
          fail(@"Dolphin PTY did not drain cleanly at replay end");
        playbackFailure = [playbackMonitor failureSnapshot];
        if (playbackFailure)
          fail([NSString stringWithFormat:@"Dolphin replay trace failed at seal: %@",
                                          playbackFailure]);
        captureFailure = [captureWriter failureSnapshot];
        if (captureFailure)
          fail([NSString stringWithFormat:@"recording content gate failed: %@",
                                          captureFailure]);
        audioEndObserved = YES;
        break;
      }
      [NSThread sleepForTimeInterval:0.020];
    }
    if (!audioEndObserved)
      fail([NSString stringWithFormat:
          @"Dolphin audio clocks did not reach a stable replay end: expected=%.6f DSP=%.6f DTK=%.6f",
          gameSeconds, lastDspSeconds, lastDtkSeconds]);
    if (!audioClockInitialized)
      fail(@"Dolphin audio clocks never initialized after the zero-audio barrier");

    // Dolphin remains suspended until the parent tears its process group down.
    // Seal exact source-frame counts so later WAV finalization cannot extend
    // the audio beyond the clock landmarks emitted here.
    int64_t sealedDspFrames = waveFramesAtPath(dspAudioPath, dspLayout);
    int64_t sealedDtkFrames = waveFramesAtPath(dtkAudioPath, dtkLayout);
    if (sealedDspFrames < dspBarrierFrames || sealedDtkFrames < dtkBarrierFrames)
      fail([NSString stringWithFormat:@"could not seal Dolphin audio clocks: DSP=%lld DTK=%lld",
                                      sealedDspFrames, sealedDtkFrames]);
    double sealedDspSeconds =
        MAX(0.0, (double)sealedDspFrames / dspLayout.sampleRate - sharedBarrierSeconds);
    double sealedDtkSeconds =
        MAX(0.0, (double)sealedDtkFrames / dtkLayout.sampleRate - sharedBarrierSeconds);
    if (sealedDspSeconds + completionTolerance < gameSeconds ||
        sealedDtkSeconds + completionTolerance < gameSeconds)
      fail([NSString stringWithFormat:
          @"sealed Dolphin audio is incomplete: expected=%.6f DSP=%.6f DTK=%.6f",
          gameSeconds, sealedDspSeconds, sealedDtkSeconds]);
    double sealedMixedAudioSeconds = MAX(sealedDspSeconds, sealedDtkSeconds);
    if (sealedMixedAudioSeconds > lastMixedAudioSeconds + 0.000001) {
      lastMixedAudioSeconds = sealedMixedAudioSeconds;
      lastMixedAudioGrowthSourceSeconds =
          MAX(videoAlignmentOffset, [captureWriter currentSourceVideoSeconds]);
    }
    lastDspSeconds = sealedDspSeconds;
    lastDtkSeconds = sealedDtkSeconds;
    NSDictionary *lastLandmark = clockLandmarks.lastObject;
    double lastLandmarkAudio = [lastLandmark[@"audio_seconds"] doubleValue];
    if (lastMixedAudioSeconds > lastLandmarkAudio + 0.000001)
      appendClockLandmark(clockLandmarks, lastMixedAudioGrowthSourceSeconds,
                          lastMixedAudioSeconds);

    terminalFinalFrozenBoundary =
        [captureWriter disarmGameplayReturningBoundary];
    terminalFinalFrozenTailProof =
        [captureWriter validateFinalFrozenTailFromAccepted:terminalAppliedBoundary
                                           throughBoundary:terminalFinalFrozenBoundary];
    if (![terminalFinalFrozenTailProof[@"passed"] boolValue])
      fail(@"terminal final frozen-tail callback proof failed");
    captureFailure = [captureWriter failureSnapshot];
    if (captureFailure)
      fail([NSString stringWithFormat:
          @"recording failed at the final frozen-tail boundary: %@",
          captureFailure]);
    CMTime stopHostTime = CMClockGetTime(CMClockGetHostTimeClock());
    [captureWriter prepareToStopAtHostTime:stopHostTime];
    dispatch_semaphore_t captureStopped = dispatch_semaphore_create(0);
    __block NSError *stopError = nil;
    [stream stopCaptureWithCompletionHandler:^(NSError *error) {
      stopError = error;
      dispatch_semaphore_signal(captureStopped);
    }];
    dispatch_semaphore_wait(captureStopped, DISPATCH_TIME_FOREVER);
    [captureWriter recordStopCompletionWithError:stopError];
    if (stopError)
      fail([NSString stringWithFormat:@"could not stop isolated-window capture: %@", stopError]);
    NSError *removeError = nil;
    if (![stream removeStreamOutput:captureWriter type:SCStreamOutputTypeScreen
                              error:&removeError])
      fail([NSString stringWithFormat:@"could not remove isolated-window stream output: %@",
                                      removeError]);
    terminalSealedStopSuffixProof =
        [captureWriter validateSealedStopSuffixFromFinalBoundary:
            terminalFinalFrozenBoundary
                                               acceptedComplete:
            terminalAppliedBoundary];
    if (![terminalSealedStopSuffixProof[@"passed"] boolValue])
      fail(@"post-boundary sealed stop suffix proof failed");
    NSError *terminalGapRecoveryError =
        [captureWriter validateTerminalGapRecoveriesFromBarrier:terminalBarrierBoundary
                                               acceptedComplete:terminalAppliedBoundary
                                        throughFinalFrozenTail:terminalFinalFrozenBoundary];
    if (terminalGapRecoveryError)
      fail([NSString stringWithFormat:
          @"terminal delivery-gap recovery failed after stream stop: %@",
          terminalGapRecoveryError]);
    [captureWriter finishWritingAtHostTime:stopHostTime];
    captureFailure = [captureWriter failureSnapshot];
    if (captureFailure)
      fail([NSString stringWithFormat:@"recording failed: %@", captureFailure]);

    NSArray<NSNumber *> *appendedPTSSequence =
        [captureWriter appendedPTSSequenceSnapshot];
    NSDictionary *postWriteAudit =
        auditEncodedVideoSamples(outputURL, appendedPTSSequence, videoAlignmentOffset);
    NSDictionary *captureDelivery =
        [captureWriter deliveryMetadataWithPostWriteAudit:postWriteAudit];
    NSDictionary *ptyMetadata = [playbackMonitor metadataSnapshot];
    if (![ptyMetadata[@"passed"] boolValue])
      fail(@"Dolphin PTY generation metadata did not pass its final gates");

    AVURLAsset *asset = [AVURLAsset URLAssetWithURL:outputURL options:nil];
    BOOL hasVideo = [asset tracksWithMediaType:AVMediaTypeVideo].count > 0;
    double duration = CMTimeGetSeconds(asset.duration);
    if (!hasVideo || !isfinite(duration) || duration <= 0.0)
      fail([NSString stringWithFormat:@"invalid captured MP4: video=%d duration=%.6f",
                                      hasVideo, duration]);
    double finalLandmarkSourceSeconds =
        [clockLandmarks.lastObject[@"source_video_seconds"] doubleValue];
    double postAudioTailSeconds = duration - finalLandmarkSourceSeconds;
    if (!isfinite(postAudioTailSeconds) || postAudioTailSeconds < tailSeconds)
      fail([NSString stringWithFormat:
          @"captured MP4 has an incomplete terminal visual tail: required=%.6f encoded=%.6f",
          tailSeconds, postAudioTailSeconds]);
    NSUInteger terminalFinalCallbackCount =
        [terminalFinalFrozenBoundary[@"callback_count"] unsignedIntegerValue];
    NSUInteger terminalAppliedCallbackCount =
        [terminalAppliedBoundary[@"callback_count"] unsignedIntegerValue];
    NSUInteger terminalAcceptedCompleteSequence =
        [terminalAppliedBoundary[@"last_complete_callback_sequence"]
            unsignedIntegerValue];
    NSUInteger terminalFinalCompleteCount =
        [terminalFinalFrozenBoundary[@"complete_sample_count"] unsignedIntegerValue];
    NSUInteger terminalAppliedCompleteCount =
        [terminalAppliedBoundary[@"complete_sample_count"] unsignedIntegerValue];
    NSUInteger terminalFinalIdleCount =
        [terminalFinalFrozenBoundary[@"idle_sample_count"] unsignedIntegerValue];
    NSUInteger terminalAppliedIdleCount =
        [terminalAppliedBoundary[@"idle_sample_count"] unsignedIntegerValue];
    BOOL terminalFinalFrozenCallbackAdvanced =
        terminalFinalCallbackCount > terminalAppliedCallbackCount;
    NSUInteger terminalFinalPostAcceptCallbackDelta =
        terminalFinalCallbackCount - terminalAcceptedCompleteSequence;
    NSUInteger terminalFinalPostSnapshotCallbackDelta =
        terminalFinalCallbackCount - terminalAppliedCallbackCount;
    NSUInteger terminalFinalCompleteDelta =
        terminalFinalCompleteCount - terminalAppliedCompleteCount;
    NSUInteger terminalFinalIdleDelta =
        terminalFinalIdleCount - terminalAppliedIdleCount;
    BOOL terminalFinalOnlyRetainedCallbacks =
        [terminalFinalFrozenTailProof[@"passed"] boolValue] &&
        terminalFinalPostSnapshotCallbackDelta ==
            terminalFinalCompleteDelta + terminalFinalIdleDelta;
    BOOL terminalFinalTransitionCountersUnchanged =
        [terminalFinalFrozenTailProof[@"visual_signature_transition_delta"]
            unsignedIntegerValue] == 0 &&
        [terminalFinalFrozenTailProof[@"content_class_transition_delta"]
            unsignedIntegerValue] == 0;
    BOOL terminalFinalVisualStateRetained =
        [terminalFinalFrozenBoundary[@"last_visual_signature"]
            isEqualToString:terminalAppliedBoundary[@"last_visual_signature"]] &&
        [terminalFinalFrozenBoundary[@"last_content_class"]
            isEqualToString:terminalAppliedBoundary[@"last_content_class"]];
    NSDictionary *startupSync = @{
      @"schema": @"idle-command-zero-audio-two-stop-inclusive-terminal-v7",
      @"method": @"idle-command-zero-audio-two-stop-inclusive-terminal-v7",
      @"requested_start_frame": @(requestedStartFrame),
      @"requested_inclusive_end": @(requestedInclusiveEndFrame),
      @"command_exclusive_end": @(commandExclusiveEndFrame),
      @"command_boundary_semantics": @"endFrame-is-exclusive-control-boundary",
      @"generation_boundary_semantics":
          @"CURRENT_FRAME-ends-at-requested-inclusive-last-content-frame",
      @"expected_content_frame_count":
          @(requestedInclusiveEndFrame - requestedStartFrame + 1),
      @"expected_trace_frame_count":
          @(requestedInclusiveEndFrame - requestedStartFrame + 1),
      @"command_install": commandInstall,
      @"pty": ptyMetadata,
      @"idle_audio_boundary": @{
        @"source_contract":
            @"idle replay accepts zero WaveFile samples; current-frame equality guard excludes opening-frame audio before the second stop",
        @"initial": @{@"dsp": initialDSPState, @"dtk": initialDTKState},
        @"first_stop": @{@"dsp": firstStopDSPState, @"dtk": firstStopDTKState},
        @"second_stop": @{@"dsp": secondStopDSPState, @"dtk": secondStopDTKState},
        @"logical_dsp_barrier_frames": @0,
        @"logical_dtk_barrier_frames": @0,
        @"visible_pcm_frames_unchanged_at_zero": @YES,
        @"passed": @YES,
      },
      @"capture_boundary": @{
        @"first_stop": firstStopBoundary,
        @"before_opening_frame_advance": beforeAdvanceBoundary,
        @"immediate_after_second_stop": secondStopImmediateBoundary,
        @"settled_after_second_stop": secondStopSettledBoundary,
        @"armed_after_opening": armedOpeningBoundary,
        @"fresh_callback_after_second_stop": @(freshCallbackAfterSecondStop),
        @"complete_sample_advanced": @(completeAdvanced),
        @"display_time_advanced": @(displayAdvanced),
        @"visual_signature_changed": @(signatureChanged),
        @"visual_signature_concrete": @(visualSignatureConcrete),
        @"video_alignment_offset_seconds": @(videoAlignmentOffset),
        @"passed": @YES,
      },
      @"stop_targets": @[@(requestedStartFrame), @(requestedStartFrame + 1)],
      @"terminal_stop_target": @(requestedInclusiveEndFrame),
      @"terminal_stop_observed": @(terminalStopObserved),
      @"terminal_apply_boundary": @{
        @"method":
            @"inclusive-current-frame-resume-exact-complete-audio-stop-retained-callback-interval-v7",
        @"at_inclusive_frame_stop": terminalStopBoundary,
        @"at_inclusive_frame_barrier": terminalBarrierBoundary,
        @"accepted_post_resume_complete": terminalAppliedBoundary,
        @"after_stopped_display_hold": terminalStoppedHoldBoundary,
        @"after_final_frozen_tail": terminalFinalFrozenBoundary,
        @"resume_host_time_mach": @(terminalResumeHostTimeMach),
        @"resume_host_time_seconds": @(terminalResumeHostTimeSeconds),
        @"maximum_application_proof_seconds":
            @(MIN(startupTimeout, TerminalApplicationProofMaximumSeconds)),
        @"accepted_complete_callback_sequence":
            terminalAppliedBoundary[@"last_complete_callback_sequence"],
        @"accepted_complete_display_time_mach":
            terminalAppliedBoundary[@"last_complete_display_time_mach"],
        @"observed_resume_to_complete_wall_seconds":
            @(terminalPostResumeWallSeconds),
        @"required_stopped_display_hold_seconds":
            @(terminalRequiredStoppedDisplayHoldSeconds),
        @"observed_stopped_display_hold_seconds":
            @(terminalStoppedDisplayHoldSeconds),
        @"pre_resume_frozen_callback_advanced":
            jsonBoolean(terminalPreResumeFrozenCallbackAdvanced),
        @"pre_resume_frozen_complete_sample_delta":
            @(terminalPreResumeFrozenCompleteDelta),
        @"pre_resume_frozen_idle_sample_delta":
            @(terminalPreResumeFrozenIdleDelta),
        @"pre_resume_frozen_only_retained_idle_or_complete":
            jsonBoolean(terminalPreResumeFrozenOnlyRetainedCallbacks),
        @"pre_resume_frozen_transition_counters_unchanged":
            jsonBoolean(terminalPreResumeFrozenTransitionCountersUnchanged),
        @"pre_resume_frozen_visual_state_retained":
            jsonBoolean(terminalPreResumeFrozenVisualStateRetained),
        @"fresh_callback_after_resume":
            jsonBoolean(terminalFreshCallbackAfterResume),
        @"complete_sample_advanced": jsonBoolean(terminalCompleteAdvanced),
        @"complete_callback_sequence_advanced":
            jsonBoolean(terminalCompleteCallbackSequenceAdvanced),
        @"complete_display_time_after_resume":
            jsonBoolean(terminalCompleteDisplayAfterResume),
        @"stopped_hold_callback_advanced":
            jsonBoolean(terminalStoppedHoldCallbackAdvanced),
        @"stopped_hold_callback_sequence_start_inclusive":
            terminalAppliedBoundary[@"last_complete_callback_sequence"],
        @"stopped_hold_callback_sequence_end_inclusive":
            terminalStoppedHoldBoundary[@"callback_count"],
        @"stopped_hold_post_accept_callback_delta":
            @(terminalStoppedHoldPostAcceptCallbackDelta),
        @"stopped_hold_post_snapshot_callback_delta":
            @(terminalStoppedHoldCallbackDelta),
        @"stopped_hold_complete_sample_delta":
            @(terminalStoppedHoldCompleteDelta),
        @"stopped_hold_complete_identity_advanced":
            jsonBoolean(terminalStoppedHoldCompleteDelta > 0),
        @"stopped_hold_idle_sample_delta":
            @(terminalStoppedHoldIdleDelta),
        @"stopped_hold_only_retained_idle_or_complete":
            jsonBoolean(terminalStoppedHoldOnlyRetainedCallbacks),
        @"stopped_hold_transition_counters_unchanged":
            jsonBoolean(terminalStoppedHoldTransitionCountersUnchanged),
        @"stopped_hold_complete_identity_consistent":
            jsonBoolean(terminalStoppedHoldCompleteIdentityConsistent),
        @"stopped_hold_callback_trace_interval": @{
          @"method": @"retained-complete-idle-callback-interval-v1",
          @"first_callback_sequence":
              terminalAppliedBoundary[@"last_complete_callback_sequence"],
          @"last_callback_sequence":
              terminalStoppedHoldBoundary[@"callback_count"],
          @"row_count": @(terminalStoppedHoldPostAcceptCallbackDelta + 1),
          @"post_accept_callback_count":
              @(terminalStoppedHoldPostAcceptCallbackDelta),
          @"complete_callback_count": @(terminalStoppedHoldCompleteDelta + 1),
          @"idle_callback_count":
              @(terminalStoppedHoldPostAcceptCallbackDelta -
                terminalStoppedHoldCompleteDelta),
          @"appended_callback_count":
              @(terminalStoppedHoldPostAcceptCallbackDelta + 1),
          @"forbidden_callback_count": @0,
          @"first_relative_display_us":
              @(llround([terminalAppliedBoundary[@"last_complete_relative_display_seconds"]
                  doubleValue] * 1000000.0)),
          @"last_relative_display_us":
              @(llround([terminalStoppedHoldBoundary[@"last_relative_display_seconds"]
                  doubleValue] * 1000000.0)),
          @"display_interval_us":
              @(llround(
                    [terminalStoppedHoldBoundary[@"last_relative_display_seconds"]
                        doubleValue] * 1000000.0) -
                llround(
                    [terminalAppliedBoundary[@"last_complete_relative_display_seconds"]
                        doubleValue] * 1000000.0)),
          @"accepted_visual_signature":
              terminalAppliedBoundary[@"last_visual_signature"],
          @"accepted_content_class":
              terminalAppliedBoundary[@"last_content_class"],
          @"visual_signature_transition_delta": @0,
          @"content_class_transition_delta": @0,
          @"passed":
              jsonBoolean(terminalStoppedHoldCallbackAdvanced &&
                          terminalStoppedHoldOnlyRetainedCallbacks &&
                          terminalStoppedHoldTransitionCountersUnchanged &&
                          terminalStoppedHoldCompleteIdentityConsistent &&
                          terminalStoppedHoldVisualStateRetained),
        },
        @"stopped_hold_display_interval_covered":
            jsonBoolean(terminalStoppedHoldDisplayIntervalCovered),
        @"stopped_hold_visual_state_retained":
            jsonBoolean(terminalStoppedHoldVisualStateRetained),
        @"final_frozen_tail_callback_advanced":
            jsonBoolean(terminalFinalFrozenCallbackAdvanced),
        @"final_frozen_tail_post_accept_callback_delta":
            @(terminalFinalPostAcceptCallbackDelta),
        @"final_frozen_tail_post_snapshot_callback_delta":
            @(terminalFinalPostSnapshotCallbackDelta),
        @"final_frozen_tail_complete_sample_delta":
            @(terminalFinalCompleteDelta),
        @"final_frozen_tail_idle_sample_delta":
            @(terminalFinalIdleDelta),
        @"final_frozen_tail_only_retained_idle_or_complete":
            jsonBoolean(terminalFinalOnlyRetainedCallbacks),
        @"final_frozen_tail_transition_counters_unchanged":
            jsonBoolean(terminalFinalTransitionCountersUnchanged),
        @"final_frozen_tail_visual_state_retained":
            jsonBoolean(terminalFinalVisualStateRetained),
        @"final_frozen_tail_callback_trace_interval":
            terminalFinalFrozenTailProof,
        @"sealed_stop_callback_suffix": terminalSealedStopSuffixProof,
        @"visible_audio_within_pre_stop_bounds":
            jsonBoolean(terminalVisibleAudioWithinBoundsBeforeParentStop),
        @"parent_stop_signal_succeeded":
            jsonBoolean(terminalParentStopSignalSucceeded),
        @"passed": jsonBoolean(terminalStopObserved &&
                                terminalPreResumeFrozenCallbackAdvanced &&
                                terminalPreResumeFrozenOnlyRetainedCallbacks &&
                                terminalPreResumeFrozenTransitionCountersUnchanged &&
                                terminalPreResumeFrozenVisualStateRetained &&
                                terminalFreshCallbackAfterResume &&
                                terminalCompleteAdvanced &&
                                terminalCompleteCallbackSequenceAdvanced &&
                                terminalCompleteDisplayAfterResume &&
                                terminalStoppedHoldCallbackAdvanced &&
                                terminalStoppedHoldOnlyRetainedCallbacks &&
                                terminalStoppedHoldTransitionCountersUnchanged &&
                                terminalStoppedHoldCompleteIdentityConsistent &&
                                terminalStoppedHoldDisplayIntervalCovered &&
                                terminalStoppedHoldVisualStateRetained &&
                                terminalFinalFrozenCallbackAdvanced &&
                                terminalFinalOnlyRetainedCallbacks &&
                                terminalFinalTransitionCountersUnchanged &&
                                terminalFinalVisualStateRetained &&
                                [terminalSealedStopSuffixProof[@"passed"] boolValue] &&
                                terminalVisibleAudioWithinBoundsBeforeParentStop &&
                                terminalParentStopSignalSucceeded),
      },
      @"terminal_audio_boundary": @{
        @"method": @"physical-wave-seal-parent-stop-no-growth-v4",
        @"audio_presentation_delay_seconds": @(audioPresentationDelay),
        @"target_content_endpoint_seconds":
            @(terminalExpectedRawAudioSeconds),
        @"finalized_buffer_allowance_seconds":
            @(TerminalAudioFinalizedBufferAllowanceSeconds),
        @"maximum_physical_overshoot_seconds":
            @(TerminalAudioMaximumPhysicalOvershootSeconds),
        @"maximum_physical_undershoot_seconds":
            @(TerminalAudioFinalizedBufferAllowanceSeconds),
        @"physical_lower_bound_seconds":
            @(terminalVisibleAudioLowerBoundSeconds),
        @"physical_upper_bound_seconds":
            @(terminalVisibleAudioUpperBoundSeconds),
        @"dsp_sample_rate": @(dspLayout.sampleRate),
        @"dtk_sample_rate": @(dtkLayout.sampleRate),
        @"dsp_physical_endpoint_seconds": @(terminalDSPEndpointSeconds),
        @"dtk_physical_endpoint_seconds": @(terminalDTKEndpointSeconds),
        @"dsp_physical_offset_seconds":
            @(terminalDSPPhysicalOffsetSeconds),
        @"dtk_physical_offset_seconds":
            @(terminalDTKPhysicalOffsetSeconds),
        @"visible_audio_within_bounds_before_parent_stop":
            jsonBoolean(terminalVisibleAudioWithinBoundsBeforeParentStop),
        @"dsp_frames_before_parent_stop":
            @(terminalDSPFramesBeforeParentStop),
        @"dtk_frames_before_parent_stop":
            @(terminalDTKFramesBeforeParentStop),
        @"dsp_frames_at_stop_baseline": @(terminalDSPFramesAtStopBaseline),
        @"dtk_frames_at_stop_baseline": @(terminalDTKFramesAtStopBaseline),
        @"stop_stability_interval_seconds":
            @(TerminalAudioStopStabilityIntervalSeconds),
        @"post_stop_observation_seconds":
            @(2.0 * TerminalAudioStopStabilityIntervalSeconds),
        @"dsp_frames_after_observation":
            @(terminalDSPFramesAfterObservation),
        @"dtk_frames_after_observation":
            @(terminalDTKFramesAfterObservation),
        @"post_stop_no_growth": jsonBoolean(terminalAudioPostStopNoGrowth),
        @"physical_pre_sigint_seal_proven": @YES,
        @"post_sigint_revalidation_and_content_seal_required": @YES,
        @"passed":
            jsonBoolean(terminalStopObserved &&
                        terminalVisibleAudioWithinBoundsBeforeParentStop &&
                        terminalAudioPostStopNoGrowth &&
                        terminalDSPFramesAtStopBaseline >=
                            terminalDSPFramesBeforeParentStop &&
                        terminalDTKFramesAtStopBaseline >=
                            terminalDTKFramesBeforeParentStop &&
                        terminalDSPEndpointSeconds >=
                            terminalVisibleAudioLowerBoundSeconds &&
                        terminalDTKEndpointSeconds >=
                            terminalVisibleAudioLowerBoundSeconds &&
                        terminalDSPEndpointSeconds <=
                            terminalVisibleAudioUpperBoundSeconds &&
                        terminalDTKEndpointSeconds <=
                            terminalVisibleAudioUpperBoundSeconds),
      },
      @"terminal_capture_tail": @{
        @"required_source_seconds": @(requiredStableSeconds),
        @"observed_source_seconds": @(terminalCaptureTailSeconds),
        @"observed_wall_seconds": @(terminalCaptureTailWallSeconds),
        @"capture_callbacks_continued":
            jsonBoolean(terminalCaptureTailSeconds >= requiredStableSeconds),
        @"passed":
            jsonBoolean(terminalCaptureTailSeconds >= requiredStableSeconds),
      },
      @"passed": @YES,
    };
    NSDictionary *metadata = @{
      @"window_id": @(target.windowID),
      @"width": @(width),
      @"height": @(height),
      @"minimum_width": @(minimumPixelWidth),
      @"minimum_height": @(minimumPixelHeight),
      @"expected_width": @(expectedPixelWidth),
      @"expected_height": @(expectedPixelHeight),
      @"window_selection_stable_polls": @4,
      @"idle_command_capture_restart": @YES,
      @"requested_capture_fps": @(requestedCaptureFPS),
      @"playback_emulation_speed": @(playbackEmulationSpeed),
      @"duration_seconds": @(duration),
      @"video_alignment_offset_seconds": @(videoAlignmentOffset),
      @"dsp_audio_barrier_frames": @(dspBarrierFrames),
      @"dtk_audio_barrier_frames": @(dtkBarrierFrames),
      @"sealed_dsp_audio_frames": @(sealedDspFrames),
      @"sealed_dtk_audio_frames": @(sealedDtkFrames),
      @"alignment_method": @"dolphin-audio-sample-barrier",
      @"completion_method": @"dolphin-audio-clock-stable-end",
      @"expected_audio_duration_seconds": @(gameSeconds),
      @"observed_audio_duration_seconds": @(lastMixedAudioSeconds),
      @"observed_dsp_duration_seconds": @(lastDspSeconds),
      @"observed_dtk_duration_seconds": @(lastDtkSeconds),
      @"completion_tolerance_seconds": @(completionTolerance),
      @"audio_end_observed": @YES,
      @"post_audio_tail_seconds": @(postAudioTailSeconds),
      @"clock_landmarks": clockLandmarks,
      @"startup_sync": startupSync,
      @"capture_delivery": captureDelivery,
      @"has_video": @YES,
      @"has_audio": @NO,
    };
    NSError *jsonError = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:metadata options:0 error:&jsonError];
    if (!json) fail([NSString stringWithFormat:@"could not encode recorder metadata: %@", jsonError]);
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
  }
  return 0;
}
