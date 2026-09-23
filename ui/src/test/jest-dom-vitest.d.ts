// Vitest 5 re-parameterised `Assertion<R, T>` and exposes `Matchers<R, T>` as
// the extension point. @testing-library/jest-dom (<= 7.0.1) still augments the
// old single-parameter `Assertion<T>`, so its matchers vanish from the types.
// Register them on the new extension point. Remove once jest-dom ships
// Vitest 5 typings.
import type { TestingLibraryMatchers } from '@testing-library/jest-dom/matchers';

declare module 'vitest' {
  // `T` must be declared for the merge to type-check even though it is unused.
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type, @typescript-eslint/no-unused-vars
  interface Matchers<R extends void | Promise<void>, T>
    extends TestingLibraryMatchers<unknown, R> {}
}
