'use strict';

import fs from 'fs/promises';

(async () => {
  process.stdout.write(
    '\n\x1B[36m[info]\x1B[0m Building component...',
  );

  const [
    ,, arg,
  ] = process.argv;

  if (!arg) {
    process.stdout.write('\n\n\x1B[31m[error]\x1B[0m You must provide a component name.\n\n');
    process.exit(0);
  }

  const dir = `./src/components/${arg}`;
  const component = `import { FC } from 'react';\nimport { ${arg}Props as Props } from './types';\nimport styles from './${arg}.module.scss';\n\nconst ${arg}: FC<Props> = () => <div className={styles.root}>${arg} Component</div>;\n\nexport default ${arg};\n`;
  const types = `export type ${arg}Props = {};\n`;
  const styles = '.root {\n  contain: content;\n}\n';
  const index = `export { default } from './${arg}';\nexport * from './types.d';\n`;
  const test = `import { render } from '@testing-library/react';\nimport ${arg} from '.';\n\ndescribe('${arg}', () => {\n  test('renders its content', () => {\n    const { container } = render(<${arg} />);\n\n    expect(container).not.toBeEmptyDOMElement();\n  });\n});\n`;

  try {
    await fs.mkdir(dir);
  } catch {
    process.stdout.write(`\n\n\x1B[33m[warn]\x1B[0m Component \x1B[36m${arg}\x1B[0m already exists.\n\n`);
    process.exit(0);
  }

  await fs.writeFile(`${dir}/index.ts`, index);
  await fs.writeFile(`${dir}/types.d.ts`, types);
  await fs.writeFile(`${dir}/${arg}.module.scss`, styles);
  await fs.writeFile(`${dir}/${arg}.test.tsx`, test);
  await fs.writeFile(`${dir}/${arg}.tsx`, component);

  process.stdout.write(
    `\n\n\x1B[32m[success]\x1B[0m Component \x1B[36m${arg}\x1B[0m created.\n\n`,
  );
})();
