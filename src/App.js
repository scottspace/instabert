// src/App.js

import React, { Component } from 'react';
import './App.css';
import Header from './components/Header';
import Feed from './components/Feed';
import {
  BrowserRouter as Router,
  Route
} from "react-router-dom";

class App extends Component {

  render() {
    return <div className="App">
      <Header />
      <section className="App-main">
        <Router>
          <Route exact path={"/"} component={Feed} />
        </Router>
      </section>
    </div>;
  };

}

export default App;
