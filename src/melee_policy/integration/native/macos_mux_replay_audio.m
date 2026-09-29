#import <AVFoundation/AVFoundation.h>
#import <AppKit/AppKit.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <QuartzCore/QuartzCore.h>
#include <math.h>

static NSString *const ContentAuditMethod =
    @"raw-source-central-gameplay-region-frame-scan-v1";
static NSString *const OutputContentAuditMethod =
    @"final-output-central-gameplay-region-frame-scan-v1";
static const double MaximumSustainedBlankSeconds = 0.500;
static const double MaximumSustainedLowMotionSeconds = 2.000;
static const double MaximumOutputJoinSampleGapSeconds = 0.100;
// At 0.5x emulation speed, one game state remains visible for about 1/30 s.
// The 40 ms ceiling admits compositor timestamp jitter around that interval
// while rejecting the roughly 1/15 s gap produced by a skipped state.
static const double MaximumSourceFrameGapSeconds = 0.040;
static const double ExpectedOutputFrameRate = 60.0;
static const double ExpectedOutputFrameIntervalSeconds = 1.0 / 60.0;
static const double OutputFrameIntervalToleranceSeconds = 0.001;
static const uint8_t NearWhiteRGBMinimum = 245;
static const uint8_t NearBlackRGBMaximum = 10;
static const double RequiredBlankPixelFraction = 0.995;
static const double MaximumLowMotionMeanLumaDelta = 0.500;

static void fail(NSString *message) {
  fprintf(stderr, "%s\n", message.UTF8String);
  exit(1);
}

static BOOL isJSONNumber(id value) {
  return [value isKindOfClass:NSNumber.class] &&
         CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID();
}

typedef NS_ENUM(NSInteger, GameplayFrameClass) {
  GameplayFrameClassActive = 0,
  GameplayFrameClassNearWhite = 1,
  GameplayFrameClassNearBlack = 2,
  GameplayFrameClassNearNeutralBlank = 3,
};

static BOOL isBlankGameplayFrameClass(GameplayFrameClass frameClass) {
  return frameClass == GameplayFrameClassNearWhite ||
         frameClass == GameplayFrameClassNearBlack ||
         frameClass == GameplayFrameClassNearNeutralBlank;
}

static NSString *blankGameplayFrameClassName(GameplayFrameClass frameClass) {
  switch (frameClass) {
    case GameplayFrameClassNearWhite:
      return @"near-white";
    case GameplayFrameClassNearBlack:
      return @"near-black";
    case GameplayFrameClassNearNeutralBlank:
      return @"near-neutral-blank";
    case GameplayFrameClassActive:
      break;
  }
  fail(@"blank-frame interval received an active gameplay frame class");
  return @"";
}

static void appendSourceBlankInterval(NSMutableArray *intervals,
                                      GameplayFrameClass frameClass,
                                      double startSeconds,
                                      double endSeconds,
                                      NSUInteger frameCount) {
  if (!isBlankGameplayFrameClass(frameClass) || !isfinite(startSeconds) ||
      !isfinite(endSeconds) || endSeconds <= startSeconds || frameCount == 0)
    fail(@"content audit produced an invalid source blank interval");
  [intervals addObject:@{
    @"class": blankGameplayFrameClassName(frameClass),
    @"source_start_seconds": @(startSeconds),
    @"source_end_seconds": @(endSeconds),
    @"source_duration_seconds": @(endSeconds - startSeconds),
    @"source_frame_count": @(frameCount),
  }];
}

static GameplayFrameClass classifyGameplayFrame(CVPixelBufferRef pixelBuffer,
                                                 NSUInteger *sampleCountOut) {
  if (!pixelBuffer) fail(@"content audit received a video sample without a pixel buffer");
  if (CVPixelBufferGetPixelFormatType(pixelBuffer) != kCVPixelFormatType_32BGRA)
    fail(@"content audit decoder did not provide 32BGRA pixels");
  CVReturn locked = CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  if (locked != kCVReturnSuccess) fail(@"content audit could not lock a decoded video frame");

  size_t width = CVPixelBufferGetWidth(pixelBuffer);
  size_t height = CVPixelBufferGetHeight(pixelBuffer);
  size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
  uint8_t *base = CVPixelBufferGetBaseAddress(pixelBuffer);
  if (!base || width < 16 || height < 16 || bytesPerRow < width * 4) {
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    fail(@"content audit decoded an invalid video frame");
  }

  // Window chrome and later burned-in labels live near the edges. This central
  // region is large enough to represent gameplay while excluding those areas.
  size_t left = width / 8;
  size_t right = width * 7 / 8;
  size_t top = height / 6;
  size_t bottom = height * 5 / 6;
  size_t stride = MAX((size_t)1, MIN(width, height) / 90);
  NSUInteger samples = 0;
  NSUInteger nearWhite = 0;
  NSUInteger nearBlack = 0;
  for (size_t y = top; y < bottom; y += stride) {
    const uint8_t *row = base + y * bytesPerRow;
    for (size_t x = left; x < right; x += stride) {
      const uint8_t *pixel = row + x * 4;
      uint8_t blue = pixel[0];
      uint8_t green = pixel[1];
      uint8_t red = pixel[2];
      nearWhite += blue >= NearWhiteRGBMinimum && green >= NearWhiteRGBMinimum &&
                   red >= NearWhiteRGBMinimum;
      nearBlack += blue <= NearBlackRGBMaximum && green <= NearBlackRGBMaximum &&
                   red <= NearBlackRGBMaximum;
      samples += 1;
    }
  }
  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  if (samples == 0) fail(@"content audit sampled no gameplay-region pixels");
  if (sampleCountOut) *sampleCountOut = samples;
  if ((double)nearWhite / (double)samples >= RequiredBlankPixelFraction)
    return GameplayFrameClassNearWhite;
  if ((double)nearBlack / (double)samples >= RequiredBlankPixelFraction)
    return GameplayFrameClassNearBlack;
  if ((double)(nearWhite + nearBlack) / (double)samples >= RequiredBlankPixelFraction)
    return GameplayFrameClassNearNeutralBlank;
  return GameplayFrameClassActive;
}

static NSData *gameplayLumaSignature(CVPixelBufferRef pixelBuffer) {
  if (!pixelBuffer) fail(@"motion audit received a video sample without a pixel buffer");
  CVReturn locked = CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  if (locked != kCVReturnSuccess) fail(@"motion audit could not lock a decoded video frame");
  size_t width = CVPixelBufferGetWidth(pixelBuffer);
  size_t height = CVPixelBufferGetHeight(pixelBuffer);
  size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
  uint8_t *base = CVPixelBufferGetBaseAddress(pixelBuffer);
  if (!base || width < 16 || height < 16 || bytesPerRow < width * 4) {
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    fail(@"motion audit decoded an invalid video frame");
  }
  const NSUInteger columns = 48;
  const NSUInteger rows = 32;
  NSMutableData *signature = [NSMutableData dataWithLength:columns * rows];
  uint8_t *values = signature.mutableBytes;
  for (NSUInteger row = 0; row < rows; ++row) {
    size_t y = height / 6 + ((2 * row + 1) * (height * 2 / 3)) / (2 * rows);
    const uint8_t *pixels = base + MIN(y, height - 1) * bytesPerRow;
    for (NSUInteger column = 0; column < columns; ++column) {
      size_t x = width / 8 + ((2 * column + 1) * (width * 3 / 4)) / (2 * columns);
      const uint8_t *pixel = pixels + MIN(x, width - 1) * 4;
      values[row * columns + column] =
          (uint8_t)((29U * pixel[0] + 150U * pixel[1] + 77U * pixel[2]) >> 8);
    }
  }
  CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
  return signature;
}

static double signatureMeanAbsoluteDelta(NSData *left, NSData *right) {
  if (!left || !right || left.length == 0 || left.length != right.length)
    return INFINITY;
  const uint8_t *leftBytes = left.bytes;
  const uint8_t *rightBytes = right.bytes;
  uint64_t total = 0;
  for (NSUInteger index = 0; index < left.length; ++index)
    total += (uint64_t)abs((int)leftBytes[index] - (int)rightBytes[index]);
  return (double)total / (double)left.length;
}

static NSDictionary *auditGameplayContent(AVURLAsset *asset, AVAssetTrack *sourceVideo,
                                          double sourceStartSeconds,
                                          double sourceEndSeconds) {
  if (!isfinite(sourceStartSeconds) || !isfinite(sourceEndSeconds) ||
      sourceStartSeconds < 0.0 || sourceEndSeconds <= sourceStartSeconds)
    fail(@"content audit source interval must be finite, nonnegative, and increasing");

  NSError *readerError = nil;
  AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:&readerError];
  if (!reader)
    fail([NSString stringWithFormat:@"could not create content audit reader: %@", readerError]);
  NSDictionary *outputSettings = @{
    (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
  };
  AVAssetReaderTrackOutput *output =
      [[AVAssetReaderTrackOutput alloc] initWithTrack:sourceVideo
                                      outputSettings:outputSettings];
  output.alwaysCopiesSampleData = NO;
  if (![reader canAddOutput:output]) fail(@"could not attach content audit video decoder");
  [reader addOutput:output];
  reader.timeRange = CMTimeRangeFromTimeToTime(
      CMTimeMakeWithSeconds(sourceStartSeconds, 60000),
      CMTimeMakeWithSeconds(sourceEndSeconds, 60000));
  if (![reader startReading])
    fail([NSString stringWithFormat:@"could not start content audit video decoder: %@",
                                    reader.error]);

  NSUInteger framesScanned = 0;
  NSUInteger nearWhiteFrames = 0;
  NSUInteger nearBlackFrames = 0;
  NSUInteger nearNeutralBlankFrames = 0;
  NSUInteger lowMotionTransitionCount = 0;
  NSUInteger minimumSampledPixels = NSUIntegerMax;
  size_t decodedWidth = 0;
  size_t decodedHeight = 0;
  double firstFrameSeconds = NAN;
  double lastFrameSeconds = NAN;
  double previousFrameSeconds = NAN;
  double maximumFrameGapSeconds = 0.0;
  GameplayFrameClass previousClass = GameplayFrameClassActive;
  double nearWhiteRunStart = NAN;
  double nearBlackRunStart = NAN;
  double nearNeutralBlankRunStart = NAN;
  double lowMotionRunStart = NAN;
  double longestNearWhiteSeconds = 0.0;
  double longestNearBlackSeconds = 0.0;
  double longestNearNeutralBlankSeconds = 0.0;
  double longestLowMotionSeconds = 0.0;
  NSData *previousSignature = nil;
  NSMutableArray *blankIntervals = [NSMutableArray array];
  double blankRunStart = NAN;
  NSUInteger blankRunFrameCount = 0;

  CMSampleBufferRef sample = NULL;
  while ((sample = [output copyNextSampleBuffer])) {
    double presentationSeconds =
        CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample));
    if (!isfinite(presentationSeconds)) {
      CFRelease(sample);
      fail(@"content audit decoded a frame with an invalid presentation timestamp");
    }
    if (presentationSeconds > sourceEndSeconds + 0.050) {
      CFRelease(sample);
      fail(@"content audit decoder exceeded its requested source interval");
    }
    // The audited gameplay range is half-open. Some decoders may return the
    // first sample at the requested end boundary, which belongs to the visual
    // tail and must not create a zero-duration gameplay interval.
    if (presentationSeconds >= sourceEndSeconds - 1e-9) {
      CFRelease(sample);
      continue;
    }
    presentationSeconds = fmin(sourceEndSeconds, fmax(sourceStartSeconds,
                                                       presentationSeconds));
    if (isfinite(previousFrameSeconds) &&
        presentationSeconds <= previousFrameSeconds + 1e-9) {
      CFRelease(sample);
      fail(@"content audit frame timestamps must increase strictly");
    }
    if (isfinite(previousFrameSeconds))
      maximumFrameGapSeconds =
          fmax(maximumFrameGapSeconds, presentationSeconds - previousFrameSeconds);

    NSUInteger sampledPixels = 0;
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sample);
    size_t frameWidth = CVPixelBufferGetWidth(pixelBuffer);
    size_t frameHeight = CVPixelBufferGetHeight(pixelBuffer);
    if (decodedWidth == 0) {
      decodedWidth = frameWidth;
      decodedHeight = frameHeight;
    } else if (frameWidth != decodedWidth || frameHeight != decodedHeight) {
      CFRelease(sample);
      fail(@"content audit decoded inconsistent source frame dimensions");
    }
    GameplayFrameClass frameClass =
        classifyGameplayFrame(pixelBuffer, &sampledPixels);
    NSData *signature = gameplayLumaSignature(pixelBuffer);
    CFRelease(sample);
    framesScanned += 1;
    minimumSampledPixels = MIN(minimumSampledPixels, sampledPixels);
    nearWhiteFrames += frameClass == GameplayFrameClassNearWhite;
    nearBlackFrames += frameClass == GameplayFrameClassNearBlack;
    nearNeutralBlankFrames += frameClass == GameplayFrameClassNearNeutralBlank;

    if (!isfinite(firstFrameSeconds)) {
      firstFrameSeconds = presentationSeconds;
      // The decoded first frame is the display state at the requested start.
      presentationSeconds = sourceStartSeconds;
    } else {
      if (previousClass == GameplayFrameClassNearWhite &&
          frameClass != GameplayFrameClassNearWhite) {
        longestNearWhiteSeconds =
            fmax(longestNearWhiteSeconds, presentationSeconds - nearWhiteRunStart);
        nearWhiteRunStart = NAN;
      } else if (previousClass != GameplayFrameClassNearWhite &&
                 frameClass == GameplayFrameClassNearWhite) {
        nearWhiteRunStart = presentationSeconds;
      }
      if (previousClass == GameplayFrameClassNearBlack &&
          frameClass != GameplayFrameClassNearBlack) {
        longestNearBlackSeconds =
            fmax(longestNearBlackSeconds, presentationSeconds - nearBlackRunStart);
        nearBlackRunStart = NAN;
      } else if (previousClass != GameplayFrameClassNearBlack &&
                 frameClass == GameplayFrameClassNearBlack) {
        nearBlackRunStart = presentationSeconds;
      }
      if (previousClass == GameplayFrameClassNearNeutralBlank &&
          frameClass != GameplayFrameClassNearNeutralBlank) {
        longestNearNeutralBlankSeconds =
            fmax(longestNearNeutralBlankSeconds,
                 presentationSeconds - nearNeutralBlankRunStart);
        nearNeutralBlankRunStart = NAN;
      } else if (previousClass != GameplayFrameClassNearNeutralBlank &&
                 frameClass == GameplayFrameClassNearNeutralBlank) {
        nearNeutralBlankRunStart = presentationSeconds;
      }
    }
    if (framesScanned == 1) {
      if (isBlankGameplayFrameClass(frameClass)) {
        blankRunStart = sourceStartSeconds;
        blankRunFrameCount = 1;
      }
    } else {
      if (isBlankGameplayFrameClass(previousClass)) {
        if (frameClass == previousClass) {
          blankRunFrameCount += 1;
        } else {
          appendSourceBlankInterval(blankIntervals, previousClass,
                                    blankRunStart, presentationSeconds,
                                    blankRunFrameCount);
          blankRunStart = NAN;
          blankRunFrameCount = 0;
        }
      }
      if (frameClass != previousClass && isBlankGameplayFrameClass(frameClass)) {
        blankRunStart = presentationSeconds;
        blankRunFrameCount = 1;
      }
    }
    if (framesScanned == 1) {
      if (frameClass == GameplayFrameClassNearWhite) nearWhiteRunStart = sourceStartSeconds;
      if (frameClass == GameplayFrameClassNearBlack) nearBlackRunStart = sourceStartSeconds;
      if (frameClass == GameplayFrameClassNearNeutralBlank)
        nearNeutralBlankRunStart = sourceStartSeconds;
    }
    if (previousSignature) {
      double meanLumaDelta = signatureMeanAbsoluteDelta(previousSignature, signature);
      if (meanLumaDelta <= MaximumLowMotionMeanLumaDelta) {
        lowMotionTransitionCount += 1;
        if (!isfinite(lowMotionRunStart)) lowMotionRunStart = previousFrameSeconds;
      } else if (isfinite(lowMotionRunStart)) {
        longestLowMotionSeconds =
            fmax(longestLowMotionSeconds, presentationSeconds - lowMotionRunStart);
        lowMotionRunStart = NAN;
      }
    }
    previousSignature = signature;
    previousClass = frameClass;
    previousFrameSeconds = presentationSeconds;
    lastFrameSeconds = presentationSeconds;
  }
  if (reader.status != AVAssetReaderStatusCompleted)
    fail([NSString stringWithFormat:@"content audit video decoder failed: %@", reader.error]);
  if (framesScanned == 0) fail(@"content audit decoded no frames in the gameplay interval");
  if (firstFrameSeconds > sourceStartSeconds + 0.100)
    fail(@"content audit did not cover the beginning of the gameplay interval");
  if (lastFrameSeconds > sourceEndSeconds + 0.001)
    fail(@"content audit decoded a frame beyond the gameplay interval");
  if (isfinite(nearWhiteRunStart))
    longestNearWhiteSeconds =
        fmax(longestNearWhiteSeconds, sourceEndSeconds - nearWhiteRunStart);
  if (isfinite(nearBlackRunStart))
    longestNearBlackSeconds =
        fmax(longestNearBlackSeconds, sourceEndSeconds - nearBlackRunStart);
  if (isfinite(nearNeutralBlankRunStart))
    longestNearNeutralBlankSeconds =
        fmax(longestNearNeutralBlankSeconds,
             sourceEndSeconds - nearNeutralBlankRunStart);
  if (isBlankGameplayFrameClass(previousClass))
    appendSourceBlankInterval(blankIntervals, previousClass, blankRunStart,
                              sourceEndSeconds, blankRunFrameCount);
  if (isfinite(lowMotionRunStart))
    longestLowMotionSeconds =
        fmax(longestLowMotionSeconds, sourceEndSeconds - lowMotionRunStart);

  if (longestNearWhiteSeconds > MaximumSustainedBlankSeconds)
    fail([NSString stringWithFormat:
        @"content audit found a sustained near-white gameplay-region interval: %.6f seconds exceeds %.6f",
        longestNearWhiteSeconds, MaximumSustainedBlankSeconds]);
  if (longestNearBlackSeconds > MaximumSustainedBlankSeconds)
    fail([NSString stringWithFormat:
        @"content audit found a sustained near-black gameplay-region interval: %.6f seconds exceeds %.6f",
        longestNearBlackSeconds, MaximumSustainedBlankSeconds]);
  if (longestNearNeutralBlankSeconds > MaximumSustainedBlankSeconds)
    fail([NSString stringWithFormat:
        @"content audit found a sustained neutral blank gameplay-region interval: %.6f seconds exceeds %.6f",
        longestNearNeutralBlankSeconds, MaximumSustainedBlankSeconds]);
  if (longestLowMotionSeconds > MaximumSustainedLowMotionSeconds)
    fail([NSString stringWithFormat:
        @"content audit found a sustained low-motion gameplay-region interval: %.6f seconds exceeds %.6f",
        longestLowMotionSeconds, MaximumSustainedLowMotionSeconds]);
  if (maximumFrameGapSeconds > MaximumSourceFrameGapSeconds)
    fail([NSString stringWithFormat:
        @"content audit found a source-frame presentation gap: %.6f seconds exceeds %.6f",
        maximumFrameGapSeconds, MaximumSourceFrameGapSeconds]);

  NSUInteger intervalNearWhiteFrames = 0;
  NSUInteger intervalNearBlackFrames = 0;
  NSUInteger intervalNearNeutralBlankFrames = 0;
  NSUInteger nearWhiteIntervalCount = 0;
  NSUInteger nearBlackIntervalCount = 0;
  NSUInteger nearNeutralBlankIntervalCount = 0;
  for (NSDictionary *interval in blankIntervals) {
    NSString *frameClass = interval[@"class"];
    NSUInteger frameCount = [interval[@"source_frame_count"] unsignedIntegerValue];
    if ([frameClass isEqualToString:@"near-white"]) {
      intervalNearWhiteFrames += frameCount;
      nearWhiteIntervalCount += 1;
    } else if ([frameClass isEqualToString:@"near-black"]) {
      intervalNearBlackFrames += frameCount;
      nearBlackIntervalCount += 1;
    } else if ([frameClass isEqualToString:@"near-neutral-blank"]) {
      intervalNearNeutralBlankFrames += frameCount;
      nearNeutralBlankIntervalCount += 1;
    } else {
      fail(@"content audit produced an unknown source blank interval class");
    }
  }
  if (intervalNearWhiteFrames != nearWhiteFrames ||
      intervalNearBlackFrames != nearBlackFrames ||
      intervalNearNeutralBlankFrames != nearNeutralBlankFrames)
    fail(@"content audit source blank interval counts differ from the frame scan");

  return @{
    @"method": ContentAuditMethod,
    @"passed": @YES,
    @"source_interval_start_seconds": @(sourceStartSeconds),
    @"source_interval_end_seconds": @(sourceEndSeconds),
    @"source_interval_duration_seconds": @(sourceEndSeconds - sourceStartSeconds),
    @"frames_scanned": @(framesScanned),
    @"first_decoded_frame_seconds": @(firstFrameSeconds),
    @"last_decoded_frame_seconds": @(lastFrameSeconds),
    @"minimum_sampled_pixels_per_frame": @(minimumSampledPixels),
    @"decoded_width": @(decodedWidth),
    @"decoded_height": @(decodedHeight),
    @"near_white_frame_count": @(nearWhiteFrames),
    @"near_black_frame_count": @(nearBlackFrames),
    @"near_neutral_blank_frame_count": @(nearNeutralBlankFrames),
    @"blank_interval_schema": @"canonical-coalesced-half-open-v1",
    @"blank_interval_time_axis": @"source-video-seconds",
    @"blank_interval_count": @(blankIntervals.count),
    @"near_white_blank_interval_count": @(nearWhiteIntervalCount),
    @"near_black_blank_interval_count": @(nearBlackIntervalCount),
    @"near_neutral_blank_interval_count": @(nearNeutralBlankIntervalCount),
    @"blank_intervals": blankIntervals,
    @"low_motion_transition_count": @(lowMotionTransitionCount),
    @"longest_near_white_interval_seconds": @(longestNearWhiteSeconds),
    @"longest_near_black_interval_seconds": @(longestNearBlackSeconds),
    @"longest_near_neutral_blank_interval_seconds": @(longestNearNeutralBlankSeconds),
    @"longest_low_motion_interval_seconds": @(longestLowMotionSeconds),
    @"maximum_frame_gap_seconds": @(maximumFrameGapSeconds),
    @"maximum_allowed_frame_gap_seconds": @(MaximumSourceFrameGapSeconds),
    @"maximum_sustained_blank_seconds": @(MaximumSustainedBlankSeconds),
    @"maximum_sustained_low_motion_seconds": @(MaximumSustainedLowMotionSeconds),
    @"maximum_low_motion_mean_luma_delta": @(MaximumLowMotionMeanLumaDelta),
    @"near_white_rgb_minimum": @(NearWhiteRGBMinimum),
    @"near_black_rgb_maximum": @(NearBlackRGBMaximum),
    @"required_blank_pixel_fraction": @(RequiredBlankPixelFraction),
    @"region": @{
      @"left_fraction": @0.125,
      @"right_fraction": @0.875,
      @"top_fraction": @(1.0 / 6.0),
      @"bottom_fraction": @(5.0 / 6.0),
    },
  };
}

static double timelineSecondsForSourceSeconds(double sourceSeconds,
                                              NSArray *clockLandmarks,
                                              NSUInteger *segmentIndexOut) {
  if (![clockLandmarks isKindOfClass:NSArray.class] || clockLandmarks.count < 2 ||
      !isfinite(sourceSeconds))
    fail(@"blank interval mapping requires finite source time and clock landmarks");
  NSDictionary *first = clockLandmarks.firstObject;
  NSDictionary *last = clockLandmarks.lastObject;
  double firstSource = [first[@"source_video_seconds"] doubleValue];
  double lastSource = [last[@"source_video_seconds"] doubleValue];
  if (sourceSeconds < firstSource - 1e-9 || sourceSeconds > lastSource + 1e-9)
    fail(@"source blank interval lies outside the replay clock landmarks");
  sourceSeconds = fmin(lastSource, fmax(firstSource, sourceSeconds));
  double firstSourceQuantized =
      CMTimeGetSeconds(CMTimeMakeWithSeconds(firstSource, 60000));
  double targetStart = 0.0;
  for (NSUInteger index = 0; index + 1 < clockLandmarks.count; ++index) {
    NSDictionary *left = clockLandmarks[index];
    NSDictionary *right = clockLandmarks[index + 1];
    double leftSource = [left[@"source_video_seconds"] doubleValue];
    double rightSource = [right[@"source_video_seconds"] doubleValue];
    double leftTimeline = [left[@"audio_seconds"] doubleValue];
    double rightTimeline = [right[@"audio_seconds"] doubleValue];
    if (!isfinite(leftSource) || !isfinite(rightSource) ||
        !isfinite(leftTimeline) || !isfinite(rightTimeline) ||
        rightSource <= leftSource || rightTimeline <= leftTimeline)
      fail(@"blank interval mapping received invalid clock landmarks");
    double targetDuration = CMTimeGetSeconds(CMTimeMakeWithSeconds(
        rightTimeline - leftTimeline, 60000));
    if (fabs(sourceSeconds - leftSource) <= 1e-9) {
      if (segmentIndexOut) *segmentIndexOut = index;
      return targetStart;
    }
    if (sourceSeconds <= rightSource + 1e-9) {
      if (segmentIndexOut) *segmentIndexOut = index;
      if (fabs(sourceSeconds - rightSource) <= 1e-9)
        return CMTimeGetSeconds(CMTimeMakeWithSeconds(
            targetStart + targetDuration, 60000));
      double sourceCoordinate = CMTimeGetSeconds(CMTimeMakeWithSeconds(
          sourceSeconds - firstSourceQuantized, 60000));
      double segmentSourceStart = CMTimeGetSeconds(CMTimeMakeWithSeconds(
          leftSource - firstSource, 60000));
      double segmentSourceDuration = CMTimeGetSeconds(CMTimeMakeWithSeconds(
          rightSource - leftSource, 60000));
      if (segmentSourceDuration <= 0.0)
        fail(@"blank interval mapping produced a zero source segment");
      double fraction = (sourceCoordinate - segmentSourceStart) /
                        segmentSourceDuration;
      fraction = fmin(1.0, fmax(0.0, fraction));
      double mapped = targetStart + fraction * targetDuration;
      return CMTimeGetSeconds(CMTimeMakeWithSeconds(mapped, 60000));
    }
    targetStart = CMTimeGetSeconds(CMTimeMakeWithSeconds(
        targetStart + targetDuration, 60000));
  }
  fail(@"blank interval mapping could not locate a clock segment");
  return NAN;
}

static NSArray *mappedSourceBlankIntervals(NSDictionary *contentAudit,
                                           NSArray *clockLandmarks) {
  NSArray *sourceIntervals = contentAudit[@"blank_intervals"];
  if (![sourceIntervals isKindOfClass:NSArray.class])
    fail(@"content audit omitted canonical source blank intervals");
  NSMutableArray *mapped = [NSMutableArray arrayWithCapacity:sourceIntervals.count];
  double previousSourceEnd = -INFINITY;
  NSString *previousClass = nil;
  for (NSUInteger index = 0; index < sourceIntervals.count; ++index) {
    NSDictionary *interval = sourceIntervals[index];
    if (![interval isKindOfClass:NSDictionary.class])
      fail(@"source blank interval entries must be objects");
    NSString *frameClass = interval[@"class"];
    NSNumber *sourceStartValue = interval[@"source_start_seconds"];
    NSNumber *sourceEndValue = interval[@"source_end_seconds"];
    NSNumber *sourceDurationValue = interval[@"source_duration_seconds"];
    NSNumber *sourceFrameCountValue = interval[@"source_frame_count"];
    if (!([frameClass isEqualToString:@"near-white"] ||
          [frameClass isEqualToString:@"near-black"] ||
          [frameClass isEqualToString:@"near-neutral-blank"]) ||
        !isJSONNumber(sourceStartValue) || !isJSONNumber(sourceEndValue) ||
        !isJSONNumber(sourceDurationValue) || !isJSONNumber(sourceFrameCountValue))
      fail(@"source blank interval has an invalid schema");
    double sourceStart = sourceStartValue.doubleValue;
    double sourceEnd = sourceEndValue.doubleValue;
    double sourceDuration = sourceDurationValue.doubleValue;
    NSInteger sourceFrameCount = sourceFrameCountValue.integerValue;
    if (!isfinite(sourceStart) || !isfinite(sourceEnd) ||
        !isfinite(sourceDuration) || sourceEnd <= sourceStart ||
        fabs(sourceDuration - (sourceEnd - sourceStart)) > 1e-9 ||
        sourceFrameCount <= 0 ||
        fabs(sourceFrameCountValue.doubleValue - (double)sourceFrameCount) > 1e-9 ||
        sourceStart < previousSourceEnd - 1e-9 ||
        ([frameClass isEqualToString:previousClass] &&
         fabs(sourceStart - previousSourceEnd) <= 1e-9))
      fail(@"source blank intervals are not canonical and coalesced");
    NSUInteger startSegment = 0;
    NSUInteger endSegment = 0;
    double timelineStart = timelineSecondsForSourceSeconds(
        sourceStart, clockLandmarks, &startSegment);
    double timelineEnd = timelineSecondsForSourceSeconds(
        sourceEnd, clockLandmarks, &endSegment);
    if (!isfinite(timelineStart) || !isfinite(timelineEnd) ||
        timelineEnd <= timelineStart)
      fail(@"source blank interval mapped to an invalid output interval");
    [mapped addObject:@{
      @"class": frameClass,
      @"source_interval_index": @(index),
      @"source_start_seconds": @(sourceStart),
      @"source_end_seconds": @(sourceEnd),
      @"source_duration_seconds": @(sourceDuration),
      @"source_frame_count": @(sourceFrameCount),
      @"timeline_start_seconds": @(timelineStart),
      @"timeline_end_seconds": @(timelineEnd),
      @"timeline_duration_seconds": @(timelineEnd - timelineStart),
      @"start_clock_segment_index": @(startSegment),
      @"end_clock_segment_index": @(endSegment),
    }];
    previousSourceEnd = sourceEnd;
    previousClass = frameClass;
  }
  return mapped;
}

static NSInteger mappedBlankIntervalIndexForFrame(NSArray *mappedIntervals,
                                                   double timelineSeconds) {
  for (NSUInteger index = 0; index < mappedIntervals.count; ++index) {
    NSDictionary *interval = mappedIntervals[index];
    double start = [interval[@"timeline_start_seconds"] doubleValue];
    double end = [interval[@"timeline_end_seconds"] doubleValue];
    if (timelineSeconds >= start && timelineSeconds < end)
      return (NSInteger)index;
  }
  return -1;
}

static NSDictionary *auditFinalOutputContent(AVURLAsset *asset, AVAssetTrack *videoTrack,
                                             double gameplayEndSeconds,
                                             NSArray *clockLandmarks,
                                             NSDictionary *contentAudit) {
  if (!isfinite(gameplayEndSeconds) || gameplayEndSeconds <= 0.0)
    fail(@"final output content audit gameplay interval must be finite and positive");
  if (![clockLandmarks isKindOfClass:NSArray.class] || clockLandmarks.count < 2)
    fail(@"final output content audit requires the replay clock landmarks");
  if (![contentAudit isKindOfClass:NSDictionary.class] ||
      ![contentAudit[@"blank_interval_schema"]
          isEqualToString:@"canonical-coalesced-half-open-v1"])
    fail(@"final output content audit requires canonical source blank intervals");
  NSArray *mappedBlankIntervals =
      mappedSourceBlankIntervals(contentAudit, clockLandmarks);

  NSError *readerError = nil;
  AVAssetReader *reader = [[AVAssetReader alloc] initWithAsset:asset error:&readerError];
  if (!reader)
    fail([NSString stringWithFormat:@"could not create final output content audit reader: %@",
                                    readerError]);
  NSDictionary *outputSettings = @{
    (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
  };
  AVAssetReaderTrackOutput *output =
      [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack
                                      outputSettings:outputSettings];
  output.alwaysCopiesSampleData = NO;
  if (![reader canAddOutput:output])
    fail(@"could not attach final output content audit video decoder");
  [reader addOutput:output];
  double assetDurationSeconds = CMTimeGetSeconds(asset.duration);
  if (!isfinite(assetDurationSeconds) || assetDurationSeconds < gameplayEndSeconds)
    fail(@"final output content audit asset is shorter than gameplay");
  // A finalized audio buffer can create an internal landmark less than one
  // video frame before gameplay ends. Decode a small amount of the attached
  // visual tail so that the sequential scan can bracket that legitimate join.
  double joinBracketingEndSeconds =
      MIN(assetDurationSeconds,
          gameplayEndSeconds + MaximumOutputJoinSampleGapSeconds);
  reader.timeRange = CMTimeRangeMake(
      kCMTimeZero, CMTimeMakeWithSeconds(joinBracketingEndSeconds, 60000));
  if (![reader startReading])
    fail([NSString stringWithFormat:@"could not start final output content audit decoder: %@",
                                    reader.error]);

  NSUInteger internalJoinCount = clockLandmarks.count - 2;
  double *beforeJoin = internalJoinCount ? calloc(internalJoinCount, sizeof(double)) : NULL;
  double *afterJoin = internalJoinCount ? calloc(internalJoinCount, sizeof(double)) : NULL;
  if (internalJoinCount && (!beforeJoin || !afterJoin))
    fail(@"could not allocate final output content audit join state");
  for (NSUInteger index = 0; index < internalJoinCount; ++index) {
    beforeJoin[index] = NAN;
    afterJoin[index] = NAN;
  }

  NSUInteger framesScanned = 0;
  NSUInteger nearWhiteFrames = 0;
  NSUInteger nearBlackFrames = 0;
  NSUInteger nearNeutralBlankFrames = 0;
  NSUInteger expectedSourceBlankFrames = 0;
  NSUInteger sourceFaithfulBlankFrames = 0;
  NSUInteger unexpectedBlankFrames = 0;
  NSUInteger missingSourceBlankFrames = 0;
  NSUInteger blankClassMismatchFrames = 0;
  NSMutableArray *blankFrameObservations = [NSMutableArray array];
  NSMutableArray *nearWhitePTS = [NSMutableArray array];
  NSMutableArray *nearBlackPTS = [NSMutableArray array];
  NSMutableArray *splitNeutralPTS = [NSMutableArray array];
  NSUInteger lowMotionTransitionCount = 0;
  NSUInteger minimumSampledPixels = NSUIntegerMax;
  size_t decodedWidth = 0;
  size_t decodedHeight = 0;
  double firstFrameSeconds = NAN;
  double lastFrameSeconds = NAN;
  double previousFrameSeconds = NAN;
  double minimumFrameGapSeconds = INFINITY;
  double maximumFrameGapSeconds = 0.0;
  double lowMotionRunStart = NAN;
  double longestLowMotionSeconds = 0.0;
  NSData *previousSignature = nil;
  CMSampleBufferRef sample = NULL;
  while ((sample = [output copyNextSampleBuffer])) {
    double presentationSeconds =
        CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample));
    if (!isfinite(presentationSeconds)) {
      CFRelease(sample);
      free(beforeJoin);
      free(afterJoin);
      fail(@"final output content audit decoded an invalid presentation timestamp");
    }
    if (presentationSeconds < -0.001 ||
        presentationSeconds > joinBracketingEndSeconds + 0.050) {
      CFRelease(sample);
      free(beforeJoin);
      free(afterJoin);
      fail(@"final output content audit decoder exceeded the gameplay interval");
    }
    presentationSeconds = fmin(joinBracketingEndSeconds,
                               fmax(0.0, presentationSeconds));
    if (isfinite(previousFrameSeconds) &&
        presentationSeconds <= previousFrameSeconds + 1e-9) {
      CFRelease(sample);
      free(beforeJoin);
      free(afterJoin);
      fail(@"final output content audit frame timestamps must increase strictly");
    }
    if (isfinite(previousFrameSeconds)) {
      double frameGapSeconds = presentationSeconds - previousFrameSeconds;
      minimumFrameGapSeconds = fmin(minimumFrameGapSeconds, frameGapSeconds);
      maximumFrameGapSeconds = fmax(maximumFrameGapSeconds, frameGapSeconds);
    }

    for (NSUInteger index = 0; index < internalJoinCount; ++index) {
      NSDictionary *point = clockLandmarks[index + 1];
      double joinSeconds = [point[@"audio_seconds"] doubleValue];
      if (presentationSeconds <= joinSeconds + 1e-9)
        beforeJoin[index] = presentationSeconds;
      if (!isfinite(afterJoin[index]) && presentationSeconds >= joinSeconds - 1e-9)
        afterJoin[index] = presentationSeconds;
    }
    if (presentationSeconds > gameplayEndSeconds + 1e-9) {
      CFRelease(sample);
      previousFrameSeconds = presentationSeconds;
      continue;
    }

    NSUInteger sampledPixels = 0;
    CVPixelBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sample);
    size_t frameWidth = CVPixelBufferGetWidth(pixelBuffer);
    size_t frameHeight = CVPixelBufferGetHeight(pixelBuffer);
    if (decodedWidth == 0) {
      decodedWidth = frameWidth;
      decodedHeight = frameHeight;
    } else if (frameWidth != decodedWidth || frameHeight != decodedHeight) {
      CFRelease(sample);
      free(beforeJoin);
      free(afterJoin);
      fail(@"final output content audit decoded inconsistent frame dimensions");
    }
    GameplayFrameClass frameClass =
        classifyGameplayFrame(pixelBuffer, &sampledPixels);
    NSData *signature = gameplayLumaSignature(pixelBuffer);
    CFRelease(sample);
    framesScanned += 1;
    minimumSampledPixels = MIN(minimumSampledPixels, sampledPixels);
    nearWhiteFrames += frameClass == GameplayFrameClassNearWhite;
    nearBlackFrames += frameClass == GameplayFrameClassNearBlack;
    nearNeutralBlankFrames += frameClass == GameplayFrameClassNearNeutralBlank;
    if (frameClass == GameplayFrameClassNearWhite)
      [nearWhitePTS addObject:@(presentationSeconds)];
    if (frameClass == GameplayFrameClassNearBlack)
      [nearBlackPTS addObject:@(presentationSeconds)];
    if (frameClass == GameplayFrameClassNearNeutralBlank)
      [splitNeutralPTS addObject:@(presentationSeconds)];
    BOOL actualBlank = isBlankGameplayFrameClass(frameClass);
    NSInteger expectedIntervalIndex =
        mappedBlankIntervalIndexForFrame(mappedBlankIntervals,
                                         presentationSeconds);
    BOOL expectedBlank = expectedIntervalIndex >= 0;
    NSString *actualClass = actualBlank
                                ? blankGameplayFrameClassName(frameClass)
                                : @"active";
    NSString *expectedClass = expectedBlank
                                  ? mappedBlankIntervals[(NSUInteger)expectedIntervalIndex][@"class"]
                                  : @"active";
    BOOL sourceFaithful = [actualClass isEqualToString:expectedClass];
    expectedSourceBlankFrames += expectedBlank;
    sourceFaithfulBlankFrames += actualBlank && sourceFaithful;
    unexpectedBlankFrames += actualBlank && !sourceFaithful;
    missingSourceBlankFrames += expectedBlank && !actualBlank;
    blankClassMismatchFrames += expectedBlank && actualBlank && !sourceFaithful;
    if (actualBlank || expectedBlank) {
      [blankFrameObservations addObject:@{
        @"timeline_seconds": @(presentationSeconds),
        @"actual_class": actualClass,
        @"expected_class": expectedClass,
        @"mapped_source_interval_index": expectedBlank
            ? @((NSUInteger)expectedIntervalIndex)
            : NSNull.null,
        @"source_faithful": @(sourceFaithful),
      }];
    }
    if (!isfinite(firstFrameSeconds)) firstFrameSeconds = presentationSeconds;
    lastFrameSeconds = presentationSeconds;
    if (previousSignature) {
      double meanLumaDelta = signatureMeanAbsoluteDelta(previousSignature, signature);
      if (meanLumaDelta <= MaximumLowMotionMeanLumaDelta) {
        lowMotionTransitionCount += 1;
        if (!isfinite(lowMotionRunStart)) lowMotionRunStart = previousFrameSeconds;
      } else if (isfinite(lowMotionRunStart)) {
        longestLowMotionSeconds =
            fmax(longestLowMotionSeconds, presentationSeconds - lowMotionRunStart);
        lowMotionRunStart = NAN;
      }
    }
    previousSignature = signature;
    previousFrameSeconds = presentationSeconds;

  }
  if (reader.status != AVAssetReaderStatusCompleted) {
    free(beforeJoin);
    free(afterJoin);
    fail([NSString stringWithFormat:@"final output content audit decoder failed: %@",
                                    reader.error]);
  }
  if (framesScanned == 0) {
    free(beforeJoin);
    free(afterJoin);
    fail(@"final output content audit decoded no gameplay frames");
  }
  if (firstFrameSeconds > 0.100) {
    free(beforeJoin);
    free(afterJoin);
    fail(@"final output content audit missed the start of gameplay");
  }
  if (!isfinite(minimumFrameGapSeconds)) {
    free(beforeJoin);
    free(afterJoin);
    fail(@"final output content audit decoded too few frames for cadence validation");
  }
  if (isfinite(lowMotionRunStart))
    longestLowMotionSeconds =
        fmax(longestLowMotionSeconds, gameplayEndSeconds - lowMotionRunStart);

  double maximumJoinSampleGapSeconds = 0.0;
  for (NSUInteger index = 0; index < internalJoinCount; ++index) {
    if (!isfinite(beforeJoin[index]) || !isfinite(afterJoin[index])) {
      free(beforeJoin);
      free(afterJoin);
      fail(@"final output content audit could not bracket every piecewise join");
    }
    maximumJoinSampleGapSeconds =
        fmax(maximumJoinSampleGapSeconds, afterJoin[index] - beforeJoin[index]);
  }
  free(beforeJoin);
  free(afterJoin);
  NSUInteger expectedFrameCount =
      (NSUInteger)ceil(gameplayEndSeconds * ExpectedOutputFrameRate - 1e-9);
  NSUInteger frameCountDelta = framesScanned > expectedFrameCount
                                   ? framesScanned - expectedFrameCount
                                   : expectedFrameCount - framesScanned;
  double minimumAllowedOutputGap =
      ExpectedOutputFrameIntervalSeconds - OutputFrameIntervalToleranceSeconds;
  double maximumAllowedOutputGap =
      ExpectedOutputFrameIntervalSeconds + OutputFrameIntervalToleranceSeconds;
  BOOL cadencePassed = frameCountDelta <= 1 &&
                       minimumFrameGapSeconds >= minimumAllowedOutputGap &&
                       maximumFrameGapSeconds <= maximumAllowedOutputGap;
  NSUInteger sourceFaithfulnessMismatchFrames =
      unexpectedBlankFrames + missingSourceBlankFrames;
  BOOL passed = sourceFaithfulnessMismatchFrames == 0 &&
                longestLowMotionSeconds <= MaximumSustainedLowMotionSeconds &&
                maximumJoinSampleGapSeconds <= MaximumOutputJoinSampleGapSeconds &&
                cadencePassed;
  return @{
    @"method": OutputContentAuditMethod,
    @"passed": @(passed),
    @"timeline_interval_start_seconds": @0.0,
    @"timeline_interval_end_seconds": @(gameplayEndSeconds),
    @"join_bracketing_interval_end_seconds": @(joinBracketingEndSeconds),
    @"join_bracketing_tail_policy": @"timestamps-only-no-content-classification",
    @"frames_scanned": @(framesScanned),
    @"first_decoded_frame_seconds": @(firstFrameSeconds),
    @"last_decoded_frame_seconds": @(lastFrameSeconds),
    @"minimum_sampled_pixels_per_frame": @(minimumSampledPixels),
    @"decoded_width": @(decodedWidth),
    @"decoded_height": @(decodedHeight),
    @"near_white_frame_count": @(nearWhiteFrames),
    @"near_black_frame_count": @(nearBlackFrames),
    @"near_neutral_blank_frame_count": @(nearNeutralBlankFrames),
    @"near_white_pts": nearWhitePTS,
    @"near_black_pts": nearBlackPTS,
    @"split_neutral_pts": splitNeutralPTS,
    @"expected_source_blank_frame_count": @(expectedSourceBlankFrames),
    @"source_faithful_blank_frame_count": @(sourceFaithfulBlankFrames),
    @"unexpected_blank_frame_count": @(unexpectedBlankFrames),
    @"missing_source_blank_frame_count": @(missingSourceBlankFrames),
    @"blank_class_mismatch_frame_count": @(blankClassMismatchFrames),
    @"source_faithfulness_mismatch_frame_count": @(sourceFaithfulnessMismatchFrames),
    @"blank_frame_observation_count": @(blankFrameObservations.count),
    @"blank_frame_observations": blankFrameObservations,
    @"source_blank_interval_count": @(mappedBlankIntervals.count),
    @"mapped_source_blank_interval_count": @(mappedBlankIntervals.count),
    @"mapped_source_blank_intervals": mappedBlankIntervals,
    @"blank_interval_mapping_method": @"clock-landmark-piecewise-linear-half-open-v1",
    @"blank_interval_mapping_quantization": @"per-segment-composition-cmtime-60000-v1",
    @"blank_interval_mapping_timescale": @60000,
    @"blank_interval_mapping_numerical_slack_seconds": @(2.0 / 60000.0),
    @"blank_interval_mapping_numerical_slack_role": @"metadata-reconciliation-only",
    @"blank_interval_membership_tolerance_seconds": @0.0,
    @"low_motion_transition_count": @(lowMotionTransitionCount),
    @"longest_low_motion_interval_seconds": @(longestLowMotionSeconds),
    @"minimum_frame_gap_seconds": @(minimumFrameGapSeconds),
    @"maximum_frame_gap_seconds": @(maximumFrameGapSeconds),
    @"minimum_allowed_frame_gap_seconds": @(minimumAllowedOutputGap),
    @"maximum_allowed_frame_gap_seconds": @(maximumAllowedOutputGap),
    @"expected_frame_rate": @(ExpectedOutputFrameRate),
    @"expected_frame_count": @(expectedFrameCount),
    @"frame_count_delta": @(frameCountDelta),
    @"frame_count_tolerance": @1,
    @"cadence_passed": @(cadencePassed),
    @"maximum_sustained_low_motion_seconds": @(MaximumSustainedLowMotionSeconds),
    @"maximum_low_motion_mean_luma_delta": @(MaximumLowMotionMeanLumaDelta),
    @"internal_piecewise_join_count": @(internalJoinCount),
    @"maximum_join_sample_gap_seconds": @(maximumJoinSampleGapSeconds),
    @"maximum_allowed_join_sample_gap_seconds": @(MaximumOutputJoinSampleGapSeconds),
    @"blank_frame_policy": @"source-faithful-near-uniform-blank-frames-over-gameplay",
  };
}

static CGFloat fittedPlayerLabelFontSize(NSString *label, CGFloat availableWidth,
                                         CGFloat maximumFontSize) {
  if (!isfinite(availableWidth) || availableWidth <= 0.0 ||
      !isfinite(maximumFontSize) || maximumFontSize < 1.0)
    fail(@"player label has invalid font-fit dimensions");
  CGFloat fontSize = maximumFontSize;
  for (NSUInteger attempt = 0; attempt < 16; ++attempt) {
    NSDictionary *attributes = @{
      NSFontAttributeName: [NSFont boldSystemFontOfSize:fontSize],
    };
    CGFloat measuredWidth = ceil([label sizeWithAttributes:attributes].width);
    if (measuredWidth <= availableWidth) return fontSize;
    fontSize = floor(fontSize * availableWidth / measuredWidth * 0.99 * 100.0) / 100.0;
    if (fontSize < 1.0) fail(@"player label cannot fit its panel legibly");
  }
  fail(@"player label font fit did not converge");
  return 0.0;
}

static void addPlayerLabel(CALayer *parent, NSString *label, CGRect frame,
                           BOOL alignRight) {
  CALayer *panel = [CALayer layer];
  panel.frame = frame;
  panel.cornerRadius = MAX(2.0, frame.size.height * 0.14);
  CGColorRef panelColor = CGColorCreateGenericRGB(0.02, 0.02, 0.02, 0.78);
  panel.backgroundColor = panelColor;
  CGColorRelease(panelColor);

  CGFloat horizontalInset = MAX(3.0, frame.size.height * 0.22);
  CGFloat textWidth = MAX(1.0, ceil(frame.size.width) - 2.0 * horizontalInset);
  CGFloat fontSize = fittedPlayerLabelFontSize(
      label, textWidth, MAX(8.0, frame.size.height * 0.45));
  NSFont *font = [NSFont boldSystemFontOfSize:fontSize];
  NSMutableParagraphStyle *paragraph = [NSMutableParagraphStyle new];
  paragraph.alignment = alignRight ? NSTextAlignmentRight : NSTextAlignmentLeft;
  paragraph.lineBreakMode = NSLineBreakByClipping;
  NSDictionary *attributes = @{
    NSFontAttributeName: font,
    NSForegroundColorAttributeName: NSColor.whiteColor,
    NSParagraphStyleAttributeName: paragraph,
  };
  NSInteger pixelWidth = MAX(1, (NSInteger)ceil(frame.size.width));
  NSInteger pixelHeight = MAX(1, (NSInteger)ceil(frame.size.height));
  NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc]
      initWithBitmapDataPlanes:NULL
                    pixelsWide:pixelWidth
                    pixelsHigh:pixelHeight
                 bitsPerSample:8
               samplesPerPixel:4
                      hasAlpha:YES
                      isPlanar:NO
                colorSpaceName:NSDeviceRGBColorSpace
                   bytesPerRow:0
                  bitsPerPixel:0];
  NSGraphicsContext *graphics = [NSGraphicsContext graphicsContextWithBitmapImageRep:bitmap];
  [NSGraphicsContext saveGraphicsState];
  [NSGraphicsContext setCurrentContext:graphics];
  [NSColor.clearColor setFill];
  NSRectFill(NSMakeRect(0.0, 0.0, pixelWidth, pixelHeight));
  NSSize textSize = [label sizeWithAttributes:attributes];
  CGFloat textY = MAX(0.0, (pixelHeight - textSize.height) / 2.0);
  [label drawInRect:NSMakeRect(horizontalInset, textY,
                              MAX(1.0, pixelWidth - 2.0 * horizontalInset),
                              textSize.height)
        withAttributes:attributes];
  [NSGraphicsContext restoreGraphicsState];

  CALayer *textImage = [CALayer layer];
  textImage.frame = panel.bounds;
  textImage.contents = (__bridge id)bitmap.CGImage;
  textImage.contentsGravity = kCAGravityResize;
  [panel addSublayer:textImage];
  [parent addSublayer:panel];
}

static AVMutableVideoComposition *labeledVideoComposition(
    AVMutableComposition *composition, NSString *playerOneLabel,
    NSString *playerTwoLabel) {
  AVMutableVideoComposition *videoComposition =
      [AVMutableVideoComposition videoCompositionWithPropertiesOfAsset:composition];
  // The convenience constructor inherits variable timing from the source
  // track. Clear that binding so frameDuration governs the exported cadence.
  videoComposition.sourceTrackIDForFrameTiming = kCMPersistentTrackID_Invalid;
  videoComposition.frameDuration = CMTimeMake(1, 60);
  CGSize size = videoComposition.renderSize;
  if (size.width < 2.0 || size.height < 2.0) fail(@"video has an invalid render size");

  CALayer *videoLayer = [CALayer layer];
  videoLayer.frame = CGRectMake(0.0, 0.0, size.width, size.height);
  CALayer *parentLayer = [CALayer layer];
  parentLayer.frame = videoLayer.frame;
  [parentLayer addSublayer:videoLayer];

  CGFloat margin = MAX(4.0, MIN(size.width, size.height) * 0.025);
  CGFloat panelHeight = MAX(12.0, MIN(56.0, size.height * 0.075));
  CGFloat availableWidth = MAX(1.0, (size.width - 3.0 * margin) / 2.0);
  CGFloat panelWidth = MIN(size.width * 0.43, availableWidth);
  CGFloat panelY = size.height - margin - panelHeight;
  addPlayerLabel(parentLayer, playerOneLabel,
                 CGRectMake(margin, panelY, panelWidth, panelHeight),
                 NO);
  addPlayerLabel(parentLayer, playerTwoLabel,
                 CGRectMake(size.width - margin - panelWidth, panelY,
                            panelWidth, panelHeight),
                 YES);
  videoComposition.animationTool =
      [AVVideoCompositionCoreAnimationTool
          videoCompositionCoreAnimationToolWithPostProcessingAsVideoLayer:videoLayer
                                                                   inLayer:parentLayer];
  return videoComposition;
}

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 8 && argc != 9) {
      fprintf(stderr, "usage: %s VIDEO.mp4 AUDIO.wav OUTPUT.mp4 VIDEO_START_SECONDS "
                      "AUDIO_DELAY_SECONDS P1_LABEL P2_LABEL [CLOCK_LANDMARKS.json]\n",
              argv[0]);
      return 2;
    }
    NSURL *videoURL = [NSURL fileURLWithPath:@(argv[1])];
    NSURL *audioURL = [NSURL fileURLWithPath:@(argv[2])];
    NSURL *outputURL = [NSURL fileURLWithPath:@(argv[3])];
    double videoStartSeconds = strtod(argv[4], NULL);
    double audioDelaySeconds = strtod(argv[5], NULL);
    NSString *playerOneLabel = @(argv[6]);
    NSString *playerTwoLabel = @(argv[7]);
    NSString *clockLandmarksPath = argc == 9 ? @(argv[8]) : nil;
    if (!isfinite(videoStartSeconds) || videoStartSeconds < 0.0)
      fail(@"video trim start must be finite and nonnegative");
    if (!isfinite(audioDelaySeconds) || audioDelaySeconds < 0.0)
      fail(@"audio presentation delay must be finite and nonnegative");
    if (playerOneLabel.length == 0 || playerTwoLabel.length == 0)
      fail(@"player labels must be nonempty");

    AVURLAsset *videoAsset = [AVURLAsset URLAssetWithURL:videoURL options:nil];
    AVURLAsset *audioAsset = [AVURLAsset URLAssetWithURL:audioURL options:nil];
    AVAssetTrack *sourceVideo = [[videoAsset tracksWithMediaType:AVMediaTypeVideo] firstObject];
    AVAssetTrack *sourceAudio = [[audioAsset tracksWithMediaType:AVMediaTypeAudio] firstObject];
    if (!sourceVideo) fail(@"isolated-window recording has no video track");
    if (!sourceAudio) fail(@"Dolphin audio dump has no audio track");

    AVMutableComposition *composition = [AVMutableComposition composition];
    AVMutableCompositionTrack *videoTrack =
        [composition addMutableTrackWithMediaType:AVMediaTypeVideo
                                 preferredTrackID:kCMPersistentTrackID_Invalid];
    AVMutableCompositionTrack *audioTrack =
        [composition addMutableTrackWithMediaType:AVMediaTypeAudio
                                 preferredTrackID:kCMPersistentTrackID_Invalid];
    NSError *error = nil;
    videoTrack.preferredTransform = sourceVideo.preferredTransform;

    BOOL piecewise = clockLandmarksPath != nil;
    NSUInteger clockLandmarkCount = 0;
    NSUInteger piecewiseSegmentCount = 0;
    NSArray *clockLandmarks = nil;
    double retimedGameDurationSeconds = 0.0;
    double rawVisualTailSeconds = 0.0;
    double videoTimelineSeconds = 0.0;
    NSDictionary *contentAudit = nil;
    double sourceVideoDurationSeconds = CMTimeGetSeconds(videoAsset.duration);
    if (!isfinite(sourceVideoDurationSeconds) || sourceVideoDurationSeconds <= 0.0)
      fail(@"isolated-window video has no positive duration");

    if (piecewise) {
      NSData *landmarkData = [NSData dataWithContentsOfFile:clockLandmarksPath];
      if (!landmarkData) fail(@"could not read replay clock landmarks");
      NSError *landmarkError = nil;
      id landmarkPayload = [NSJSONSerialization JSONObjectWithData:landmarkData
                                                            options:0
                                                              error:&landmarkError];
      if (![landmarkPayload isKindOfClass:NSDictionary.class])
        fail([NSString stringWithFormat:@"clock landmark payload must be an object: %@",
                                        landmarkError]);
      NSArray *landmarks = ((NSDictionary *)landmarkPayload)[@"clock_landmarks"];
      if (![landmarks isKindOfClass:NSArray.class] || landmarks.count < 2)
        fail(@"piecewise replay mux requires at least two clock landmarks");
      clockLandmarkCount = landmarks.count;
      clockLandmarks = landmarks;

      double previousSource = -1.0;
      double previousAudio = -1.0;
      for (NSUInteger index = 0; index < landmarks.count; ++index) {
        NSDictionary *point = landmarks[index];
        if (![point isKindOfClass:NSDictionary.class])
          fail(@"clock landmark entries must be objects");
        NSNumber *sourceValue = point[@"source_video_seconds"];
        NSNumber *audioValue = point[@"audio_seconds"];
        if (!isJSONNumber(sourceValue) || !isJSONNumber(audioValue))
          fail(@"clock landmark coordinates must be numbers");
        double source = sourceValue.doubleValue;
        double audio = audioValue.doubleValue;
        if (!isfinite(source) || !isfinite(audio) || source < 0.0 || audio < 0.0)
          fail(@"clock landmark coordinates must be finite and nonnegative");
        if (index == 0) {
          if (fabs(source - videoStartSeconds) > 0.050 || fabs(audio) > 0.001)
            fail(@"first clock landmark differs from the synchronized media barrier");
        } else if (source <= previousSource || audio <= previousAudio) {
          fail(@"clock landmarks must increase strictly on both axes");
        }
        if (source > sourceVideoDurationSeconds + 0.001)
          fail(@"clock landmark exceeds the raw video duration");
        previousSource = source;
        previousAudio = audio;
      }

      contentAudit = auditGameplayContent(
          videoAsset, sourceVideo,
          [landmarks.firstObject[@"source_video_seconds"] doubleValue],
          [landmarks.lastObject[@"source_video_seconds"] doubleValue]);

      NSDictionary *firstPoint = landmarks.firstObject;
      NSDictionary *lastPoint = landmarks.lastObject;
      double firstSourceSeconds = [firstPoint[@"source_video_seconds"] doubleValue];
      double lastSourceSeconds = [lastPoint[@"source_video_seconds"] doubleValue];
      retimedGameDurationSeconds = [lastPoint[@"audio_seconds"] doubleValue];
      rawVisualTailSeconds = sourceVideoDurationSeconds - lastSourceSeconds;
      if (rawVisualTailSeconds <= 0.0)
        fail(@"piecewise replay video has no post-audio terminal visual tail");

      // Insert one continuous source range so every adjacent frame shares a
      // media boundary. Scaling in reverse keeps each earlier source-relative
      // range stable while shifting the already-retimed suffix as one block.
      CMTime continuousSourceStart = CMTimeMakeWithSeconds(firstSourceSeconds, 60000);
      CMTime continuousSourceDuration =
          CMTimeSubtract(videoAsset.duration, continuousSourceStart);
      error = nil;
      if (![videoTrack insertTimeRange:CMTimeRangeMake(continuousSourceStart,
                                                       continuousSourceDuration)
                               ofTrack:sourceVideo
                                atTime:kCMTimeZero
                                 error:&error])
        fail([NSString stringWithFormat:@"could not insert continuous replay video: %@",
                                        error]);

      for (NSInteger index = (NSInteger)landmarks.count - 2; index >= 0; --index) {
        NSDictionary *left = landmarks[index];
        NSDictionary *right = landmarks[index + 1];
        double sourceStartSeconds = [left[@"source_video_seconds"] doubleValue];
        double sourceEndSeconds = [right[@"source_video_seconds"] doubleValue];
        double targetDurationSeconds = [right[@"audio_seconds"] doubleValue] -
                                       [left[@"audio_seconds"] doubleValue];
        CMTime sourceRelativeStart =
            CMTimeMakeWithSeconds(sourceStartSeconds - firstSourceSeconds, 60000);
        CMTime sourceDuration =
            CMTimeMakeWithSeconds(sourceEndSeconds - sourceStartSeconds, 60000);
        CMTime destinationDuration =
            CMTimeMakeWithSeconds(targetDurationSeconds, 60000);
        [videoTrack scaleTimeRange:CMTimeRangeMake(sourceRelativeStart, sourceDuration)
                        toDuration:destinationDuration];
      }
      piecewiseSegmentCount = landmarks.count - 1;
      videoTimelineSeconds = retimedGameDurationSeconds + rawVisualTailSeconds;
    } else {
      CMTime start = CMTimeMakeWithSeconds(videoStartSeconds, 60000);
      CMTime availableVideo = CMTimeSubtract(videoAsset.duration, start);
      double availableVideoSeconds = CMTimeGetSeconds(availableVideo);
      if (!isfinite(availableVideoSeconds) || availableVideoSeconds <= 0.0)
        fail(@"isolated-window video has no positive duration after trimming");
      if (![videoTrack insertTimeRange:CMTimeRangeMake(start, availableVideo)
                               ofTrack:sourceVideo
                                atTime:kCMTimeZero
                                 error:&error])
        fail([NSString stringWithFormat:@"could not trim isolated-window video: %@", error]);
      videoTimelineSeconds = availableVideoSeconds;
      retimedGameDurationSeconds = availableVideoSeconds;
      contentAudit = auditGameplayContent(videoAsset, sourceVideo, videoStartSeconds,
                                          sourceVideoDurationSeconds);
    }

    CMTime audioDelay = CMTimeMakeWithSeconds(audioDelaySeconds, 60000);
    double audioAssetDurationSeconds = CMTimeGetSeconds(audioAsset.duration);
    if (!isfinite(audioAssetDurationSeconds) || audioAssetDurationSeconds <= 0.0)
      fail(@"Dolphin audio has no positive duration");
    CMTime maximumAudioDuration =
        CMTimeMakeWithSeconds(MAX(0.0, videoTimelineSeconds - audioDelaySeconds), 60000);
    CMTime audioDuration = piecewise ? audioAsset.duration
                                     : CMTimeMinimum(maximumAudioDuration, audioAsset.duration);
    double audioDurationSeconds = CMTimeGetSeconds(audioDuration);
    BOOL audioFullyPreserved =
        isfinite(audioDurationSeconds) &&
        fabs(audioDurationSeconds - audioAssetDurationSeconds) <= 0.001;
    if (piecewise &&
        fabs(retimedGameDurationSeconds - audioAssetDurationSeconds) > 0.001)
      fail([NSString stringWithFormat:
          @"final replay clock landmark differs from sealed Dolphin audio: landmark=%.6f audio=%.6f",
          retimedGameDurationSeconds, audioAssetDurationSeconds]);
    if (piecewise && !audioFullyPreserved)
      fail(@"piecewise replay mux did not retain the complete Dolphin audio asset");
    if (piecewise && audioDelaySeconds + audioDurationSeconds > videoTimelineSeconds + 0.001)
      fail(@"terminal visual tail is too short to preserve the complete delayed audio track");
    error = nil;
    if (![audioTrack insertTimeRange:CMTimeRangeMake(kCMTimeZero, audioDuration)
                             ofTrack:sourceAudio
                              atTime:audioDelay
                               error:&error])
      fail([NSString stringWithFormat:@"could not add Dolphin audio: %@", error]);

    [[NSFileManager defaultManager] removeItemAtURL:outputURL error:nil];
    AVAssetExportSession *exporter =
        [[AVAssetExportSession alloc] initWithAsset:composition
                                         presetName:AVAssetExportPresetHighestQuality];
    if (!exporter) fail(@"could not create MP4 export session");
    exporter.outputURL = outputURL;
    exporter.outputFileType = AVFileTypeMPEG4;
    exporter.shouldOptimizeForNetworkUse = YES;
    exporter.videoComposition =
        labeledVideoComposition(composition, playerOneLabel, playerTwoLabel);
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [exporter exportAsynchronouslyWithCompletionHandler:^{ dispatch_semaphore_signal(done); }];
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    if (exporter.status != AVAssetExportSessionStatusCompleted)
      fail([NSString stringWithFormat:@"MP4 export failed: %@", exporter.error]);

    AVURLAsset *outputAsset = [AVURLAsset URLAssetWithURL:outputURL options:nil];
    AVAssetTrack *outputVideo =
        [[outputAsset tracksWithMediaType:AVMediaTypeVideo] firstObject];
    BOOL hasVideo = outputVideo != nil;
    BOOL hasAudio = [outputAsset tracksWithMediaType:AVMediaTypeAudio].count > 0;
    double outputDuration = CMTimeGetSeconds(outputAsset.duration);
    if (!hasVideo || !hasAudio || !isfinite(outputDuration) || outputDuration <= 0.0)
      fail([NSString stringWithFormat:@"invalid final MP4: video=%d audio=%d duration=%.6f",
                                      hasVideo, hasAudio, outputDuration]);
    double measuredOutputFrameRate = outputVideo.nominalFrameRate;
    if (!isfinite(measuredOutputFrameRate) ||
        fabs(measuredOutputFrameRate - ExpectedOutputFrameRate) > 0.001)
      fail([NSString stringWithFormat:
          @"final MP4 track frame rate differs from %.3f fps: %.6f",
          ExpectedOutputFrameRate, measuredOutputFrameRate]);
    double audioEndSeconds = audioDelaySeconds + audioDurationSeconds;
    double visualTailSeconds = fmax(0.0, videoTimelineSeconds - audioEndSeconds);
    if (fabs(outputDuration - videoTimelineSeconds) > 2.0 / 60.0 + 0.001)
      fail([NSString stringWithFormat:@"final MP4 duration differs from the visual timeline: output=%.6f video=%.6f",
                                      outputDuration, videoTimelineSeconds]);
    NSArray *outputAuditLandmarks = clockLandmarks ?: @[
      @{
        @"source_video_seconds": @(videoStartSeconds),
        @"audio_seconds": @0.0,
      },
      @{
        @"source_video_seconds": @(sourceVideoDurationSeconds),
        @"audio_seconds": @(retimedGameDurationSeconds),
      },
    ];
    NSDictionary *outputContentAudit = auditFinalOutputContent(
        outputAsset, outputVideo, retimedGameDurationSeconds,
        outputAuditLandmarks, contentAudit);
    if (![outputContentAudit[@"passed"] boolValue]) {
      BOOL preserveRejectedOutput =
          [[NSProcessInfo.processInfo.environment
              objectForKey:@"MELEE_POLICY_PRESERVE_REJECTED_MUX"]
              isEqualToString:@"1"];
      if (!preserveRejectedOutput)
        [[NSFileManager defaultManager] removeItemAtURL:outputURL error:nil];
      fail([NSString stringWithFormat:
          @"final output content audit rejected synthesized blank frames or uncovered joins: "
           "source=%@ output=%@ preserved=%d",
          contentAudit, outputContentAudit, preserveRejectedOutput]);
    }
    NSDictionary *metadata = @{
      @"duration_seconds": @(outputDuration),
      @"has_video": @YES,
      @"has_audio": @YES,
      @"video_trim_start_seconds": @(videoStartSeconds),
      @"video_duration_seconds": @(videoTimelineSeconds),
      @"output_frame_rate": @(ExpectedOutputFrameRate),
      @"measured_output_frame_rate": @(measuredOutputFrameRate),
      @"audio_presentation_delay_seconds": @(audioDelaySeconds),
      @"audio_inserted_duration_seconds": @(audioDurationSeconds),
      @"audio_end_seconds": @(audioEndSeconds),
      @"visual_tail_after_audio_seconds": @(visualTailSeconds),
      @"timing_method": piecewise ? @"dolphin-audio-clock-piecewise" : @"start-trim-only",
      @"clock_landmark_count": @(clockLandmarkCount),
      @"piecewise_segment_count": @(piecewiseSegmentCount),
      @"retimed_game_duration_seconds": @(retimedGameDurationSeconds),
      @"raw_visual_tail_seconds": @(rawVisualTailSeconds),
      @"audio_fully_preserved": @(audioFullyPreserved),
      @"labels_burned_in": @YES,
      @"player_labels": @[ playerOneLabel, playerTwoLabel ],
      @"content_audit": contentAudit,
      @"output_content_audit": outputContentAudit,
    };
    NSError *jsonError = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:metadata options:0 error:&jsonError];
    if (!json) fail([NSString stringWithFormat:@"could not encode mux metadata: %@", jsonError]);
    fwrite(json.bytes, 1, json.length, stdout);
    fputc('\n', stdout);
  }
  return 0;
}
