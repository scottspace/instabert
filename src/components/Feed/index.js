// src/components/contacts.js
import React, { Component } from "react";
import Post from '../Post';
import LoadingIndicator from '../LoadingIndicator';
import { trackPromise } from 'react-promise-tracker';

class Feed extends Component {

  state = {
    articles: [],
    query: ''
  };

  componentDidMount() {
    var q = this.props.location.search;
    var url = 'https://us-central1-octo-news.cloudfunctions.net/articles';
    if (q.length > 0) {
      url += q;
    }    
    this.setState({'query': q});
    trackPromise(
      fetch(url)
        .then(res => res.json())
        .then((data) => {
          this.setState({ articles: data })
        })
        .catch(console.log)
    )
  };

  render() {
    console.log("Feed object", this.props);
    return (
      <div>
        {this.state.articles.map((article) => (
          <Post article={article} />
        ))}
        <LoadingIndicator />
      </div>
    )
  };
}

export default Feed;
