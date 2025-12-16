'use strict';

import open from 'open';
import dotenv from 'dotenv';

(async () => {
  process.stdout.write('\n\x1B[36m[info]\x1B[0m Getting credentials...');
  dotenv.config();

  const { SPACE, TOKEN } = process.env;

  process.stdout.write('\n\n\x1B[36m[info]\x1B[0m Opening graphiql in your default browser...');
  await open(`https://graphql.contentful.com/content/v1/spaces/${SPACE}/explore?access_token=${TOKEN}`);
  process.stdout.write('\n\n\x1B[36m[info]\x1B[0m Done.\n\n');
})();
