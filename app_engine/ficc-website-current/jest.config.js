module.exports = {
  clearMocks: true,
  moduleFileExtensions: [
    "js",
    "json",
    "ts",
    "tsx",
  ],
  moduleNameMapper: {
    '\\.(css|less|sass|scss)$': 'identity-obj-proxy',
    '@components/(.*)$': '<rootDir>/src/components/$1',
    '@pages/(.*)$': '<rootDir>/pages/$1',
    '@src/(.*)$': '<rootDir>/src/$1',
    '#constants': '<rootDir>/src/constants/index.ts',
    '#lib': '<rootDir>/src/lib/index.ts',
    '_contentful': '<rootDir>/src/contentful/index.ts',
    '#graphql': '<rootDir>/src/graphql/index.ts',
  },
  roots: [
    "<rootDir>"
  ],
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  testMatch: [
    "**/*.test.tsx"
  ],
  testPathIgnorePatterns: [
    "/node_modules/",
    "/.yarn/",
    "/.next/",
    "/public/",
    "/styles/",
    "/scripts/",
    "/out/",
  ],
  timers: "fake",
  transform: {
    '^.+\\.(ts|tsx)$': 'babel-jest',
  },
  transformIgnorePatterns: [
    "/node_modules/"
  ],
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname',
  ],
  testEnvironment: 'jsdom',
};
