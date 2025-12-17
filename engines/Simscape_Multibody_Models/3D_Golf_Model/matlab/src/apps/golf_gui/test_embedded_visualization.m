function test_embedded_visualization()
% TEST_EMBEDDED_VISUALIZATION - Automated test script for embedded SkeletonPlotter
%
% This script performs automated tests on the embedded visualization feature
% and provides a report of results.
%
% Tests:
%   1. App launches successfully
%   2. Tab 3 loads with embedded visualization
%   3. Visualization is visible and functional
%   4. Playback controls respond
%   5. Cleanup works properly
%
% Usage:
%   test_embedded_visualization()

fprintf('\n');
fprintf('==============================================================\n');
fprintf('  Embedded Visualization Test Suite\n');
fprintf('==============================================================\n');
fprintf('\n');

test_results = struct();
test_count = 0;
pass_count = 0;

%% Test 1: Launch Application
test_count = test_count + 1;
fprintf('Test %d: Launching application...\n', test_count);
try
    app_handles = launch_tabbed_app();
    pause(2); % Allow time for GUI to render

    if ishandle(app_handles.main_fig) && isvalid(app_handles.main_fig)
        fprintf('  ✓ PASS: Application launched successfully\n');
        test_results.test1 = 'PASS';
        pass_count = pass_count + 1;
    else
        fprintf('  ✗ FAIL: Application figure is invalid\n');
        test_results.test1 = 'FAIL';
        return;
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    test_results.test1 = 'FAIL';
    return;
end

%% Test 2: Tab 3 Initialization
test_count = test_count + 1;
fprintf('\nTest %d: Checking Tab 3 initialization...\n', test_count);
try
    if isfield(app_handles, 'tab3_handles')
        tab3 = app_handles.tab3_handles;

        if isfield(tab3, 'viz_panel') && ishandle(tab3.viz_panel)
            fprintf('  ✓ PASS: Tab 3 visualization panel exists\n');
            test_results.test2 = 'PASS';
            pass_count = pass_count + 1;
        else
            fprintf('  ✗ FAIL: Visualization panel not found\n');
            test_results.test2 = 'FAIL';
        end
    else
        fprintf('  ✗ FAIL: Tab 3 handles not found\n');
        test_results.test2 = 'FAIL';
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    test_results.test2 = 'FAIL';
end

%% Test 3: Embedded Content Check
test_count = test_count + 1;
fprintf('\nTest %d: Checking for embedded visualization content...\n', test_count);
try
    % Check if panel has children (axes, controls, etc.)
    panel_children = findobj(tab3.viz_panel, '-depth', 1);

    if length(panel_children) > 5  % Should have many children (axes, panels, buttons)
        fprintf('  ✓ PASS: Panel has %d child objects (expected many)\n', length(panel_children));
        test_results.test3 = 'PASS';
        pass_count = pass_count + 1;

        % List some key components
        axes_found = findobj(panel_children, 'Type', 'axes');
        panels_found = findobj(panel_children, 'Type', 'uipanel');
        buttons_found = findobj(panel_children, 'Type', 'uicontrol', 'Style', 'pushbutton');

        fprintf('    - Found %d axes\n', length(axes_found));
        fprintf('    - Found %d panels\n', length(panels_found));
        fprintf('    - Found %d buttons\n', length(buttons_found));
    else
        fprintf('  ✗ FAIL: Panel has only %d children (expected many more)\n', length(panel_children));
        test_results.test3 = 'FAIL';
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    test_results.test3 = 'FAIL';
end

%% Test 4: 3D Axes Rendering
test_count = test_count + 1;
fprintf('\nTest %d: Checking 3D visualization axes...\n', test_count);
try
    all_axes = findobj(tab3.viz_panel, 'Type', 'axes');

    if ~isempty(all_axes)
        main_ax = all_axes(1); % Assume first is main 3D axes

        % Check if it has 3D content
        view_angles = get(main_ax, 'View');
        children = get(main_ax, 'Children');

        if length(children) > 10  % Should have many objects (cylinders, spheres, quivers)
            fprintf('  ✓ PASS: 3D axes has %d graphical objects\n', length(children));
            fprintf('    - View angles: [%.1f, %.1f]\n', view_angles(1), view_angles(2));
            test_results.test4 = 'PASS';
            pass_count = pass_count + 1;
        else
            fprintf('  ✗ FAIL: 3D axes has only %d objects (expected more)\n', length(children));
            test_results.test4 = 'FAIL';
        end
    else
        fprintf('  ✗ FAIL: No axes found in visualization panel\n');
        test_results.test4 = 'FAIL';
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    test_results.test4 = 'FAIL';
end

%% Test 5: Control Elements Present
test_count = test_count + 1;
fprintf('\nTest %d: Checking for control elements...\n', test_count);
try
    % Find key controls
    play_button = findobj(tab3.viz_panel, 'Type', 'uicontrol', 'Style', 'togglebutton', 'String', 'Play');
    sliders = findobj(tab3.viz_panel, 'Type', 'uicontrol', 'Style', 'slider');
    dropdowns = findobj(tab3.viz_panel, 'Type', 'uicontrol', 'Style', 'popupmenu');

    controls_found = 0;
    if ~isempty(play_button)
        fprintf('    ✓ Play button found\n');
        controls_found = controls_found + 1;
    else
        fprintf('    ✗ Play button NOT found\n');
    end

    if length(sliders) >= 3
        fprintf('    ✓ Sliders found (%d)\n', length(sliders));
        controls_found = controls_found + 1;
    else
        fprintf('    ✗ Expected 3+ sliders, found %d\n', length(sliders));
    end

    if ~isempty(dropdowns)
        fprintf('    ✓ Dropdown menu found\n');
        controls_found = controls_found + 1;
    else
        fprintf('    ✗ Dropdown menu NOT found\n');
    end

    if controls_found >= 2
        fprintf('  ✓ PASS: Key controls present (%d/3)\n', controls_found);
        test_results.test5 = 'PASS';
        pass_count = pass_count + 1;
    else
        fprintf('  ✗ FAIL: Missing key controls (only %d/3 found)\n', controls_found);
        test_results.test5 = 'FAIL';
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    test_results.test5 = 'FAIL';
end

%% Test 6: No Separate Window Created
test_count = test_count + 1;
fprintf('\nTest %d: Verifying no separate window was created...\n', test_count);
try
    % Find all figures
    all_figs = findall(0, 'Type', 'figure');

    % Count figures with "Golf Swing Plotter" in name
    skeleton_figs = [];
    for i = 1:length(all_figs)
        fig_name = get(all_figs(i), 'Name');
        if contains(fig_name, 'Golf Swing Plotter')
            skeleton_figs = [skeleton_figs; all_figs(i)];
        end
    end

    if isempty(skeleton_figs)
        fprintf('  ✓ PASS: No separate SkeletonPlotter window found (embedded correctly)\n');
        test_results.test6 = 'PASS';
        pass_count = pass_count + 1;
    else
        fprintf('  ✗ FAIL: Found %d separate SkeletonPlotter window(s)\n', length(skeleton_figs));
        fprintf('    This suggests embedding is not working correctly\n');
        test_results.test6 = 'FAIL';
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    test_results.test6 = 'FAIL';
end

%% Test 7: Cleanup Test
test_count = test_count + 1;
fprintf('\nTest %d: Testing cleanup...\n', test_count);
fprintf('  NOTE: Will close application in 3 seconds...\n');
pause(3);

try
    % Close the app
    if ishandle(app_handles.main_fig)
        close(app_handles.main_fig);
        pause(1); % Allow time for cleanup
    end

    % Check if properly cleaned up
    if ~ishandle(app_handles.main_fig)
        fprintf('  ✓ PASS: Application closed successfully\n');
        test_results.test7 = 'PASS';
        pass_count = pass_count + 1;
    else
        fprintf('  ✗ FAIL: Application figure still exists\n');
        test_results.test7 = 'FAIL';
    end
catch ME
    fprintf('  ✗ FAIL: %s\n', ME.message);
    test_results.test7 = 'FAIL';
end

%% Summary
fprintf('\n');
fprintf('==============================================================\n');
fprintf('  Test Summary\n');
fprintf('==============================================================\n');
fprintf('  Total Tests: %d\n', test_count);
fprintf('  Passed:      %d\n', pass_count);
fprintf('  Failed:      %d\n', test_count - pass_count);
fprintf('  Success Rate: %.1f%%\n', (pass_count/test_count)*100);
fprintf('==============================================================\n');
fprintf('\n');

if pass_count == test_count
    fprintf('✓ ALL TESTS PASSED! Embedded visualization is working correctly.\n');
else
    fprintf('⚠ SOME TESTS FAILED. Review the results above for details.\n');
end

fprintf('\n');

end
