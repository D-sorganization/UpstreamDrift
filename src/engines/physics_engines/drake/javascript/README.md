# JavaScript/TypeScript Source Code

This directory contains JavaScript and TypeScript source code for the project.

## Structure

```
javascript/
├── src/          # Source code
├── tests/        # Test files
└── config/       # Configuration files (webpack, babel, etc.)
```

## Setup

### Node.js Project

```bash
# Initialize npm project
npm init -y

# Install dependencies
npm install

# Install development dependencies
npm install --save-dev jest eslint prettier typescript @types/node
```

### TypeScript Setup

```bash
# Initialize TypeScript
npx tsc --init

# Install TypeScript
npm install --save-dev typescript @types/node
```

## Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

## Linting

```bash
# Run ESLint
npm run lint

# Fix ESLint issues
npm run lint -- --fix
```

## Building

```bash
# Build TypeScript
npm run build

# Build for production
npm run build:prod
```

## Configuration Files

- `package.json` - Node.js dependencies and scripts
- `tsconfig.json` - TypeScript configuration
- `.eslintrc.js` - ESLint configuration
- `.prettierrc` - Prettier configuration
- `jest.config.js` - Jest test configuration
