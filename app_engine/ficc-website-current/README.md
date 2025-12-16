# ficc.ai Website

[![Netlify Status](https://api.netlify.com/api/v1/badges/5dd7ae2b-1769-4587-8842-44c49e292d1a/deploy-status)](https://app.netlify.com/sites/elastic-mirzakhani-faf6a6/deploys)

## Getting Started

This project uses:

- [Yarn 2](https://yarnpkg.com/getting-started/migration)
- [Node 14](https://nodejs.org/en/) or higher
- [Typescript 4](https://www.typescriptlang.org/) or higher
- [Next](https://nextjs.org/)
- [GraphQL](https://graphql.org/)
- [Jest](https://jestjs.io/) with [React Testing Library](https://testing-library.com/docs/react-testing-library/intro/)

### Build Setup

Install packages

```bash
yarn
```

Start the project on a local server

```bash
yarn start
```

Build for production and launch server
```bash
yarn build

yarn serve
```

Generate static HTML build
```bash
yarn static
```

### Dev Commands

Create a new component

```bash
yarn component <ComponentName>
```

Launch GraphiQL in your browser

```bash
yarn graphiql
```

## Contributing

### Accessibility

This project follows [WCAG 2.1](https://www.w3.org/TR/WCAG21/) levels A and AA guidelines for accessibility. Please write compliant code.

### Typescript

Please refrain from using the `any` type, unless the type _truly_ accepts anything. If a type is not yet known, please use the `unknown` type and then assert the appropriate type.

### Testing

Please avoid writing tests that cover implementation details of a component, that includes coverage. Just focus on what the user should expect to be presented with and to interact with. [Read more](https://kentcdodds.com/blog/testing-implementation-details)

## Recommended Extensions for VS Code

- ESLint from Dirk Baeumer
- GraphQL from GraphQL Foundation
