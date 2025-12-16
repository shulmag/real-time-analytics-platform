import { render } from '@testing-library/react';
import Section from '.';

describe('Section', () => {
  test('renders its content', () => {
    const { container } = render(<Section />);

    expect(container).not.toBeEmptyDOMElement();
  });
});
