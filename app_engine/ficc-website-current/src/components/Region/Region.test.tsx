import { render } from '@testing-library/react';
import Region from '.';

describe('Region', () => {
  test('renders its content', () => {
    const { container } = render(<Region />);

    expect(container).not.toBeEmptyDOMElement();
  });
});
