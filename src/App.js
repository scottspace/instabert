// src/App.js

import React, { Component } from 'react';
import './App.css';
import Header from './components/Header';
import Post from './components/Post';
import Feed from './components/Feed';

class App extends Component {

  state = {
    articles: []
  };

  componentDidMount() {
    fetch('https://us-central1-octo-news.cloudfunctions.net/articles')
      .then(res => res.json())
      .then((data) => {
        this.setState({ articles: data })
      })
      .catch(console.log)
  };

  render() {
    return <div className="App">
      <Header />
      <section className="App-main">
        <Feed articles={this.state.articles} />
      </section>
    </div>;
  };

}

export default App;
