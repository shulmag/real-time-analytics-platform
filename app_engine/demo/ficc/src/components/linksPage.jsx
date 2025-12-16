import React from "react";
import { Link } from "react-router-dom";

function LinksPage() {

  return (
    <React.Fragment>
      <div>
        <h1>Demo Component Sandbox</h1>
      </div>

      <Link to="/hello">A very simple functional component</Link>
      <br />
      <Link to="/demoMain">Demo Prototype</Link>
      <br />
      <Link to="/searchForm">Search Form</Link>
      <br />
      <Link to="/Graph">Curve</Link>
    </React.Fragment>
  );
}

export default LinksPage;
