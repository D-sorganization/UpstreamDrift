export declare const MIN_NODE_MAJOR: number;
export declare function assertNodeVersion(version: string): void;
export declare function buildSteps(options: {
  install: boolean;
}): [string, string[]][];
export declare function launchCommand(port: number): [string, string[]];
