import { render } from '@testing-library/react';
import Aside from '.';

describe('Aside', () => {
  test('renders its content', () => {
    const { container } = render(<Aside />);

    expect(container).not.toBeEmptyDOMElement();
  });
});
