// src/App.js

import React, { Component } from 'react';
import './App.css';
//import Header from './components/Header';
import Feed from './components/Feed';
import HeatMap from './components/HeatMap';

import {
  BrowserRouter as Router,
  Route
} from "react-router-dom";

class App extends Component {

  render() {
    return <div className="App">
      <section className="App-main">
        <Router>
          <Route exact path={"/"} component={Feed} />
          <Route exact path={"/map"} component={HeatMap} />
        </Router>
      </section>
    </div>;
  };
}

export default App;
