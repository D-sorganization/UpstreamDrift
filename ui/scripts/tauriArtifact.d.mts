export declare const ARTIFACT_DIR: string;
interface App {
  productName: string;
  version: string;
  commit: string;
}
export declare function artifactName(file: string, app: App): string;
export declare function assertSameVersion(
  appVersion: string,
  packageVersion: string,
): void;
export declare function tauriBuildCommand(): [string, string[]];
export declare function packageArtifacts(
  options: App & { bundleDir: string; outDir: string },
): { name: string; path: string }[];
