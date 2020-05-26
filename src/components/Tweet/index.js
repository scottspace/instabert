import "./Tweet.css";
import React, { Component } from "react";

class Tweet extends Component {

  render() {
    const article = this.props.article;
    const nickname = article.who;
    //const avatar = article.favicon;
    const image = article.image;
    const caption = article.title + " " + article.snippet;
    const topic = article.topic_text;
    const emo = 3+Math.min(3,Math.max(-3,article.z));
    //const city = article.city.replace(' ','+');
    const emote = ['😭','😢','🙁','😐','🙂','😍','🥳'];
    var smiley = emote[emo]; 

    return (
    <div className="tweet">
      <a href={article.url}>
        <img alt={caption} src={image} />
      </a>
      <div className="title">
        {article.title}
      </div>
      <div className="topic">
      {smiley} {topic} <br/> {nickname}
      </div>
    </div>
    );
    }
};

export default Tweet;