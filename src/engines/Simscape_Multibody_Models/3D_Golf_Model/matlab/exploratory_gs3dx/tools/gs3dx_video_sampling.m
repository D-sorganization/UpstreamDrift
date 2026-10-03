function [frames, stride, fps] = gs3dx_video_sampling(rate_hz, n_frames, requested_stride, requested_frames)
%GS3DX_VIDEO_SAMPLING Uniform capture samples with physical playback timing.
% The last video sample need not be the last capture frame. A constant-rate
% video cannot truthfully represent irregularly spaced capture samples.
    arguments
        rate_hz (1,1) double
        n_frames (1,1) double
        requested_stride (1,1) double = 0
        requested_frames (1,:) double = []
    end
    assert(isfinite(rate_hz) && rate_hz > 0, 'gs3dx:video_sampling', 'Capture rate must be positive and finite');
    assert(isfinite(n_frames) && n_frames >= 1 && n_frames == fix(n_frames), ...
        'gs3dx:video_sampling', 'Capture size must be a positive integer');
    assert(isfinite(requested_stride) && requested_stride >= 0 && requested_stride == fix(requested_stride), ...
        'gs3dx:video_sampling', 'Stride must be zero (automatic) or a positive integer');
    if isempty(requested_frames)
        stride = requested_stride;
        if stride == 0
            stride = max(1, round(rate_hz / 30));
        end
        frames = 1:stride:n_frames;
    else
        frames = requested_frames;
        assert(all(isfinite(frames) & frames >= 1 & frames <= n_frames & frames == fix(frames)), ...
            'gs3dx:video_sampling', 'Frames must be integer capture indices');
        spacing = diff(frames);
        assert(isempty(spacing) || (spacing(1) > 0 && all(spacing == spacing(1))), ...
            'gs3dx:video_sampling', 'Video frames must be strictly increasing and uniformly spaced');
        stride = max(1, requested_stride);
        if ~isempty(spacing)
            stride = spacing(1);
        end
    end
    fps = rate_hz / stride;
end
