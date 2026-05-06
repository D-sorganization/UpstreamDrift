function tests = test_frame_search_checkpoint
%TEST_FRAME_SEARCH_CHECKPOINT Unit tests for the frame_search package.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
thisFile = mfilename('fullpath');
matlabDir = fileparts(fileparts(thisFile));
addpath(matlabDir);
testCase.TestData.matlabDir = matlabDir;
end

function setup(testCase)
testCase.TestData.runDir = tempname;
mkdir(testCase.TestData.runDir);
end

function teardown(testCase)
if exist(testCase.TestData.runDir, 'dir')
    rmdir(testCase.TestData.runDir, 's');
end
end

function state = makeState(sha, lastFrame)
state = struct( ...
    'manifest_sha256', sha, ...
    'last_frame_idx', lastFrame, ...
    'previous_torque', [1, 2, 3], ...
    'current_state', struct('frame_index', lastFrame), ...
    'committed_torques', repmat([1, 2, 3], lastFrame, 1), ...
    'frame_scores', (1:lastFrame)', ...
    'wall_clock_per_frame', 0.5 * ones(lastFrame, 1));
end

function test_checkpoint_writes_every_k_frames(testCase)
runDir = testCase.TestData.runDir;
state5 = makeState('abc123', 5);
frame_search.checkpoint(runDir, state5);
chkPath = fullfile(runDir, 'checkpoint.mat');
verifyTrue(testCase, isfile(chkPath));

% Overwrite at frame 10 — atomic replace must keep it valid.
state10 = makeState('abc123', 10);
frame_search.checkpoint(runDir, state10);
loaded = load(chkPath);
verifyEqual(testCase, loaded.last_frame_idx, 10);
verifyEqual(testCase, size(loaded.committed_torques, 1), 10);
end

function test_checkpoint_rejects_missing_field(testCase)
bad = struct('manifest_sha256', 'x', 'last_frame_idx', 1);
verifyError(testCase, @() frame_search.checkpoint(testCase.TestData.runDir, bad), ...
    'frame_search:checkpoint:MissingField');
end

function test_resume_continues_from_last_frame(testCase)
runDir = testCase.TestData.runDir;
state = makeState('hash-A', 7);
frame_search.checkpoint(runDir, state);

[loaded, resumed] = frame_search.resume(runDir, 'hash-A');
verifyTrue(testCase, resumed);
verifyEqual(testCase, loaded.last_frame_idx, 7);
verifyEqual(testCase, loaded.previous_torque, [1, 2, 3]);
end

function test_resume_rejects_sha_mismatch(testCase)
runDir = testCase.TestData.runDir;
state = makeState('hash-A', 4);
frame_search.checkpoint(runDir, state);

warnState = warning('off', 'frame_search:resume:ShaMismatch');
cleanup = onCleanup(@() warning(warnState));
[~, resumed] = frame_search.resume(runDir, 'hash-B');
verifyFalse(testCase, resumed);
end

function test_resume_no_checkpoint_returns_fresh(testCase)
[~, resumed] = frame_search.resume(testCase.TestData.runDir, 'sha');
verifyFalse(testCase, resumed);
end

function test_stale_lock_detected(testCase)
runDir = testCase.TestData.runDir;
state = makeState('sha', 3);
frame_search.checkpoint(runDir, state);
progressCsv = fullfile(runDir, 'progress.csv');
fid = fopen(progressCsv, 'w');
fprintf(fid, 'frame_idx,selected_candidate,score,wall_clock_s,timestamp\n');
fprintf(fid, '1,0,0.1,0.5,2026-05-05T00:00:01\n');
fclose(fid);

% Backdate progress.csv mtime far into the past.
oldTime = now - 1; %#ok<TNOW1>
java.io.File(progressCsv).setLastModified(int64((oldTime - 719529) * 86400 * 1000));

[~, lastWarn] = lastwarn('', '');
warnState = warning('on', 'frame_search:resume:StaleLock');
cleanup = onCleanup(@() warning(warnState));
lastwarn('');
[~, resumed] = frame_search.resume(runDir, 'sha', 1.0, 2.0);
[msg, id] = lastwarn();
verifyTrue(testCase, resumed);
verifyEqual(testCase, id, 'frame_search:resume:StaleLock');
verifyNotEmpty(testCase, msg);
end
