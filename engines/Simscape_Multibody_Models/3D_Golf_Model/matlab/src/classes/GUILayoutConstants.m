classdef GUILayoutConstants
    % GUILAYOUTCONSTANTS Standard layout dimensions for Golf Analysis GUIs
    %
    % This class defines consistent sizing and spacing for GUI components
    % to ensure professional, user-friendly interfaces across all applications.
    %
    % Usage:
    %   layout = GUILayoutConstants.getDefaultLayout();
    %   figWidth = min(layout.FIGURE_MAX_WIDTH, screenSize(3) * layout.SCREEN_WIDTH_RATIO);
    %
    % Design Philosophy:
    %   - Responsive: Adapts to screen size while respecting maximums
    %   - Consistent: Standard spacing creates visual rhythm
    %   - Accessible: Minimum sizes ensure touch-friendly targets
    %   - Professional: Based on common GUI design guidelines
    %
    % References:
    %   - Apple Human Interface Guidelines
    %   - Material Design spacing system (8px base unit)
    %   - WCAG 2.1 touch target size guidelines (44x44 px minimum)
    %
    % See also: UICOLORS, DATASET_GUI

    properties (Constant)
        %% Figure Dimensions

        % FIGURE_MAX_WIDTH - Maximum figure width in pixels
        % Prevents window from being too wide on large displays
        % Value: 1800 px (suitable for 1920x1080 screens with taskbar)
        FIGURE_MAX_WIDTH = 1800

        % FIGURE_MAX_HEIGHT - Maximum figure height in pixels
        % Prevents window from being too tall
        % Value: 1000 px (allows for OS taskbar and window chrome)
        FIGURE_MAX_HEIGHT = 1000

        % SCREEN_WIDTH_RATIO - Fraction of screen width to use
        % Ensures window doesn't completely fill screen
        % Value: 0.9 (90% - leaves 5% margin on each side)
        SCREEN_WIDTH_RATIO = 0.9

        % SCREEN_HEIGHT_RATIO - Fraction of screen height to use
        % Value: 0.9 (90% - leaves margin for taskbar/dock)
        SCREEN_HEIGHT_RATIO = 0.9

        %% Title Bar

        % TITLE_HEIGHT_RATIO - Title bar height as fraction of figure
        % Value: 0.06 (6% of figure height, ~60px for 1000px window)
        TITLE_HEIGHT_RATIO = 0.06

        % TITLE_FONT_SIZE - Font size for main title
        % Value: 14 pt (readable, professional)
        TITLE_FONT_SIZE = 14

        %% Buttons

        % BUTTON_WIDTH_RATIO - Standard button width as fraction of figure
        % Value: 0.07 (7% of figure width, ~126px for 1800px window)
        BUTTON_WIDTH_RATIO = 0.07

        % BUTTON_HEIGHT_RATIO - Standard button height as fraction of title bar
        % Value: 0.6 (60% of title bar height, ~36px)
        BUTTON_HEIGHT_RATIO = 0.6

        % BUTTON_Y_RATIO - Vertical position in title bar
        % Value: 0.2 (20% from bottom, centers 60% height button)
        BUTTON_Y_RATIO = 0.2

        % BUTTON_SPACING_RATIO - Space between buttons as fraction of figure
        % Value: 0.01 (1% of figure width, ~18px)
        BUTTON_SPACING_RATIO = 0.01

        % BUTTON_MIN_WIDTH - Minimum button width in pixels
        % Ensures touch-friendly targets per WCAG guidelines
        % Value: 44 px (minimum touch target size)
        BUTTON_MIN_WIDTH = 44

        % BUTTON_MIN_HEIGHT - Minimum button height in pixels
        % Value: 44 px (minimum touch target size)
        BUTTON_MIN_HEIGHT = 44

        %% Panels & Spacing

        % PANEL_PADDING_RATIO - Padding inside panels
        % Value: 0.01 (1% of figure dimension)
        PANEL_PADDING_RATIO = 0.01

        % ELEMENT_SPACING_RATIO - Vertical spacing between elements
        % Value: 0.005 (0.5% of figure height, ~5px for 1000px window)
        ELEMENT_SPACING_RATIO = 0.005

        % GROUP_SPACING_RATIO - Spacing between groups of elements
        % Value: 0.02 (2% of figure height, larger than element spacing)
        GROUP_SPACING_RATIO = 0.02

        % TAB_CONTENT_PADDING - Padding inside tab content
        % Value: 10 px (fixed, based on 8px grid system)
        TAB_CONTENT_PADDING = 10

        %% Text & Labels

        % LABEL_FONT_SIZE - Font size for labels
        % Value: 10 pt (standard label size)
        LABEL_FONT_SIZE = 10

        % TEXT_FONT_SIZE - Font size for body text
        % Value: 9 pt (slightly smaller than labels)
        TEXT_FONT_SIZE = 9

        % HEADING_FONT_SIZE - Font size for section headings
        % Value: 12 pt (larger than body text)
        HEADING_FONT_SIZE = 12

        %% Input Fields

        % EDIT_FIELD_HEIGHT - Height of text edit fields in pixels
        % Value: 25 px (comfortable for text entry)
        EDIT_FIELD_HEIGHT = 25

        % EDIT_FIELD_MIN_WIDTH - Minimum width of edit fields
        % Value: 60 px (enough for 4-5 digit numbers)
        EDIT_FIELD_MIN_WIDTH = 60

        % DROPDOWN_HEIGHT - Height of dropdown menus
        % Value: 25 px (matches edit field height)
        DROPDOWN_HEIGHT = 25

        %% Control Panel

        % CONTROL_PANEL_WIDTH_RATIO - Width of side control panel
        % Value: 0.25 (25% of figure width)
        CONTROL_PANEL_WIDTH_RATIO = 0.25

        % CONTROL_PANEL_MIN_WIDTH - Minimum control panel width
        % Value: 200 px (ensures usability)
        CONTROL_PANEL_MIN_WIDTH = 200

        %% Tabs

        % TAB_HEIGHT - Height of tab bar in pixels
        % Value: 30 px (standard tab height)
        TAB_HEIGHT = 30

        % TAB_MIN_WIDTH - Minimum tab width in pixels
        % Value: 80 px (enough for tab label)
        TAB_MIN_WIDTH = 80
    end

    methods (Static)
        function layout = getDefaultLayout()
            % GETDEFAULTLAYOUT Returns complete layout constants as struct
            %
            % Returns:
            %   layout - Struct containing all layout constants
            %
            % Example:
            %   layout = GUILayoutConstants.getDefaultLayout();
            %   figWidth = min(layout.FIGURE_MAX_WIDTH, ...
            %                  screenSize(3) * layout.SCREEN_WIDTH_RATIO);

            layout = struct();

            % Figure dimensions
            layout.FIGURE_MAX_WIDTH = GUILayoutConstants.FIGURE_MAX_WIDTH;
            layout.FIGURE_MAX_HEIGHT = GUILayoutConstants.FIGURE_MAX_HEIGHT;
            layout.SCREEN_WIDTH_RATIO = GUILayoutConstants.SCREEN_WIDTH_RATIO;
            layout.SCREEN_HEIGHT_RATIO = GUILayoutConstants.SCREEN_HEIGHT_RATIO;

            % Title bar
            layout.TITLE_HEIGHT_RATIO = GUILayoutConstants.TITLE_HEIGHT_RATIO;
            layout.TITLE_FONT_SIZE = GUILayoutConstants.TITLE_FONT_SIZE;

            % Buttons
            layout.BUTTON_WIDTH_RATIO = GUILayoutConstants.BUTTON_WIDTH_RATIO;
            layout.BUTTON_HEIGHT_RATIO = GUILayoutConstants.BUTTON_HEIGHT_RATIO;
            layout.BUTTON_Y_RATIO = GUILayoutConstants.BUTTON_Y_RATIO;
            layout.BUTTON_SPACING_RATIO = GUILayoutConstants.BUTTON_SPACING_RATIO;
            layout.BUTTON_MIN_WIDTH = GUILayoutConstants.BUTTON_MIN_WIDTH;
            layout.BUTTON_MIN_HEIGHT = GUILayoutConstants.BUTTON_MIN_HEIGHT;

            % Panels & Spacing
            layout.PANEL_PADDING_RATIO = GUILayoutConstants.PANEL_PADDING_RATIO;
            layout.ELEMENT_SPACING_RATIO = GUILayoutConstants.ELEMENT_SPACING_RATIO;
            layout.GROUP_SPACING_RATIO = GUILayoutConstants.GROUP_SPACING_RATIO;
            layout.TAB_CONTENT_PADDING = GUILayoutConstants.TAB_CONTENT_PADDING;

            % Text & Labels
            layout.LABEL_FONT_SIZE = GUILayoutConstants.LABEL_FONT_SIZE;
            layout.TEXT_FONT_SIZE = GUILayoutConstants.TEXT_FONT_SIZE;
            layout.HEADING_FONT_SIZE = GUILayoutConstants.HEADING_FONT_SIZE;

            % Input Fields
            layout.EDIT_FIELD_HEIGHT = GUILayoutConstants.EDIT_FIELD_HEIGHT;
            layout.EDIT_FIELD_MIN_WIDTH = GUILayoutConstants.EDIT_FIELD_MIN_WIDTH;
            layout.DROPDOWN_HEIGHT = GUILayoutConstants.DROPDOWN_HEIGHT;

            % Control Panel
            layout.CONTROL_PANEL_WIDTH_RATIO = GUILayoutConstants.CONTROL_PANEL_WIDTH_RATIO;
            layout.CONTROL_PANEL_MIN_WIDTH = GUILayoutConstants.CONTROL_PANEL_MIN_WIDTH;

            % Tabs
            layout.TAB_HEIGHT = GUILayoutConstants.TAB_HEIGHT;
            layout.TAB_MIN_WIDTH = GUILayoutConstants.TAB_MIN_WIDTH;
        end

        function px = ratioToPixels(ratio, dimension)
            % RATIOTOPIXELS Convert ratio to pixels for given dimension
            %
            % Args:
            %   ratio - Ratio value (0-1)
            %   dimension - Reference dimension in pixels
            %
            % Returns:
            %   px - Calculated pixels
            %
            % Example:
            %   buttonWidth = GUILayoutConstants.ratioToPixels(0.07, 1800);
            %   % Returns: 126 pixels

            arguments
                ratio (1,1) double {mustBeInRange(ratio, 0, 1)}
                dimension (1,1) double {mustBePositive}
            end

            px = round(ratio * dimension);
        end
    end
end
