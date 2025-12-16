import {
  Fragment, FC, useState,
} from 'react';
import Markdown, { MarkdownToJSX } from 'markdown-to-jsx';
import { state } from '@providers/StateProvider';
import { createSlug, createSpan } from '#hooks';
import { cls } from '#utils';
import styles from './Section.module.scss';

import type { SectionProps as Props, FormDataType } from './types';


const copyOpts: MarkdownToJSX.Options = {
  forceBlock: true,
  wrapper: Fragment,
};

const headingOpts: MarkdownToJSX.Options = {
  wrapper: 'h3',
  forceWrapper: true,
};

const Section: FC<Props> = ({
  heading,
  subHeading,
  copy,
  parent,
  icon,
  contactForm,
}) => {
  const { slug } = state();
  const title: string = createSpan(heading);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [emailSent, setEmailSent] = useState(false);

  const encode = (data: FormDataType) => Object.keys(data)
    .map(
      key => `${encodeURIComponent(key)}=${encodeURIComponent(data[key])}`,
    )
    .join('&');


  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setEmailSent(true);
    fetch('/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: encode({
        'form-name': 'contact-form',
        name,
        email,
      }),
    })
      .catch(error => alert(error));
    setName('');
    setEmail('');
  };

  return (
    <section
      id={createSlug(heading)}
      className={cls(
        styles.root,
        parent && styles.parent,
        slug !== '/' && styles.secondary,
      )}
    >
      <div data-wrap={parent ? true : undefined}>
        {icon && (
          <picture>
            <img
              src="/img/ficc-icons-sprite.svg"
              alt=""
            />
          </picture>
        )}
        {parent ? (
          <h2>{heading}</h2>
        ) : (
          <Markdown options={headingOpts}>{title}</Markdown>
        )}
        {subHeading && <h4>{subHeading}</h4>}
        {copy && <Markdown options={copyOpts}>{copy}</Markdown>}
        {contactForm && (
          <form
            className={styles.form}
            method="POST"
            data-netlify="true"
            name="contact-form"
            onSubmit={handleSubmit}
          >
            <input type="hidden"
              name="form-name"
              value="contact-form"
            />
            <input
              type="hidden"
              name="subject"
              value={`Contact form submission from ${name}`}
            />
            <label htmlFor="name"
              className={styles.hidden}
            >Name</label>
            <input type="name"
              name="name"
              placeholder="Name"
              onChange={e => setName(e.target.value)}
              value={name}
              required
            />
            <label htmlFor="email"
              className={styles.hidden}
            >Email</label>
            <input type="email"
              name="email"
              placeholder="Email Address"
              onChange={e => setEmail(e.target.value)}
              value={email}
              required
            />
            <button type="submit"
              disabled={emailSent}
              className={emailSent ? styles.sent : ''}
            >{emailSent ? 'Thank You' : 'Submit'}</button>
          </form>
        )}
      </div>
    </section>
  );
};


export default Section;
