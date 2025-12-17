classdef UIColors
    % UICOLORS Standard color palette for Golf Analysis GUI applications
    %
    % This class defines a consistent, professional color scheme based on
    % Material Design principles with modifications for scientific visualization.
    %
    % Usage:
    %   colors = UIColors.getColorScheme();
    %   figure('Color', colors.background);
    %   uicontrol('BackgroundColor', colors.primary);
    %
    % Color Philosophy:
    %   - Primary: Professional blue (trust, stability)
    %   - Success: Green (positive outcomes, completion)
    %   - Danger: Red (errors, warnings requiring attention)
    %   - Warning: Amber (caution, important information)
    %   - Neutrals: Grays (backgrounds, borders, text)
    %
    % References:
    %   - Material Design Color System: https://material.io/design/color
    %   - WCAG 2.1 AA contrast ratios for accessibility
    %
    % See also: DATASET_GUI, GOLFSWINGVISUALIZER

    properties (Constant)
        %% Primary Brand Colors

        % PRIMARY - Sharp blue (Blue 700)
        % Used for: Headers, primary buttons, active states
        % RGB: [0.2, 0.4, 0.8] = #3366CC
        % Contrast ratio vs white: 4.8:1 (WCAG AA compliant)
        PRIMARY = [0.2, 0.4, 0.8]

        % SECONDARY - Bright blue (Blue 500)
        % Used for: Highlights, secondary buttons, links
        % RGB: [0.3, 0.5, 0.9] = #4D7FE6
        SECONDARY = [0.3, 0.5, 0.9]

        %% Status Colors

        % SUCCESS - Sharp green (Green 600)
        % Used for: Success messages, completion indicators, positive values
        % RGB: [0.2, 0.7, 0.3] = #33B34D
        SUCCESS = [0.2, 0.7, 0.3]

        % DANGER - Sharp red (Red 600)
        % Used for: Error messages, delete buttons, critical warnings
        % RGB: [0.8, 0.2, 0.2] = #CC3333
        DANGER = [0.8, 0.2, 0.2]

        % WARNING - Sharp amber (Amber 600)
        % Used for: Warning messages, caution indicators
        % RGB: [0.9, 0.6, 0.1] = #E6991A
        WARNING = [0.9, 0.6, 0.1]

        %% Background Colors

        % BACKGROUND - Slightly cooler light gray
        % Used for: Main application background
        % RGB: [0.95, 0.95, 0.97] = #F2F2F7
        % Provides subtle blue tint for reduced eye strain
        BACKGROUND = [0.95, 0.95, 0.97]

        % PANEL - Pure white
        % Used for: Content panels, cards, elevated surfaces
        % RGB: [1, 1, 1] = #FFFFFF
        PANEL = [1, 1, 1]

        %% Text Colors

        % TEXT - Very dark gray (87% opacity black)
        % Used for: Primary text, headings
        % RGB: [0.1, 0.1, 0.1] = #1A1A1A
        % Contrast ratio vs white: 15.3:1 (WCAG AAA compliant)
        TEXT = [0.1, 0.1, 0.1]

        % TEXT_LIGHT - Darker medium gray (60% opacity black)
        % Used for: Secondary text, labels, disabled text
        % RGB: [0.4, 0.4, 0.4] = #666666
        % Contrast ratio vs white: 4.6:1 (WCAG AA compliant)
        TEXT_LIGHT = [0.4, 0.4, 0.4]

        %% Border & UI Element Colors

        % BORDER - Darker gray border
        % Used for: Borders, dividers, outlines
        % RGB: [0.8, 0.8, 0.8] = #CCCCCC
        BORDER = [0.8, 0.8, 0.8]

        % TAB_ACTIVE - Bright blue (Blue 200)
        % Used for: Active tab indicator
        % RGB: [0.7, 0.8, 1.0] = #B3CCFF
        TAB_ACTIVE = [0.7, 0.8, 1.0]

        % TAB_INACTIVE - Light gray
        % Used for: Inactive tabs
        % RGB: [0.9, 0.9, 0.9] = #E6E6E6
        TAB_INACTIVE = [0.9, 0.9, 0.9]

        % LIGHT_GREY - Light grey
        % Used for: Main text buttons, neutral UI elements
        % RGB: [0.85, 0.85, 0.85] = #D9D9D9
        LIGHT_GREY = [0.85, 0.85, 0.85]
    end

    methods (Static)
        function colors = getColorScheme()
            % GETCOLORSCHEME Returns complete color scheme as struct
            %
            % Returns:
            %   colors - Struct containing all color constants
            %
            % Example:
            %   colors = UIColors.getColorScheme();
            %   figure('Color', colors.background);

            colors = struct();
            colors.primary = UIColors.PRIMARY;
            colors.secondary = UIColors.SECONDARY;
            colors.success = UIColors.SUCCESS;
            colors.danger = UIColors.DANGER;
            colors.warning = UIColors.WARNING;
            colors.background = UIColors.BACKGROUND;
            colors.panel = UIColors.PANEL;
            colors.text = UIColors.TEXT;
            colors.textLight = UIColors.TEXT_LIGHT;
            colors.border = UIColors.BORDER;
            colors.tabActive = UIColors.TAB_ACTIVE;
            colors.tabInactive = UIColors.TAB_INACTIVE;
            colors.lightGrey = UIColors.LIGHT_GREY;
        end

        function hex = toHex(rgb)
            % TOHEX Convert RGB array to hex color string
            %
            % Args:
            %   rgb - [R, G, B] array with values 0-1
            %
            % Returns:
            %   hex - Hex color string (e.g., '#3366CC')
            %
            % Example:
            %   hex = UIColors.toHex(UIColors.PRIMARY);
            %   % Returns: '#3366CC'

            arguments
                rgb (1,3) double {mustBeInRange(rgb, 0, 1)}
            end

            r = dec2hex(round(rgb(1) * 255), 2);
            g = dec2hex(round(rgb(2) * 255), 2);
            b = dec2hex(round(rgb(3) * 255), 2);
            hex = ['#', r, g, b];
        end

        function rgb = fromHex(hex)
            % FROMHEX Convert hex color string to RGB array
            %
            % Args:
            %   hex - Hex color string (e.g., '#3366CC' or '3366CC')
            %
            % Returns:
            %   rgb - [R, G, B] array with values 0-1
            %
            % Example:
            %   rgb = UIColors.fromHex('#3366CC');
            %   % Returns: [0.2, 0.4, 0.8]

            arguments
                hex (1,:) char
            end

            % Remove '#' if present
            if hex(1) == '#'
                hex = hex(2:end);
            end

            r = hex2dec(hex(1:2)) / 255;
            g = hex2dec(hex(3:4)) / 255;
            b = hex2dec(hex(5:6)) / 255;
            rgb = [r, g, b];
        end
    end
end
