import "./Post.css";

/**
 * {"date":"0",
 * "index":"0",
 * "doc_id":"720",
 * "topic":"0",
 * "who":"newscientist.com",
 * "url":"https://www.newscientist.com/article/2237475-coronavirus-latest-no-new-deaths-in-china-and-hopes-of-plateau-in-nyc/",
 * "title":"Coronavirus latest: No new deaths in China and hopes of plateau in NYC",
 * "image":"https://images.newscientist.com/wp-content/uploads/2020/04/07165837/gettyimages-1217375799_web.jpg",
 * "distance":2.807396411895752,
 * "snippet":"Scientists detected a drop in nitrogen dioxide emissions over several cities including Paris, Milan and Madrid.",
 * "topic_text":"county toll deaths cases"}
 * 
 */

import React, { Component } from "react";
class Post extends Component {
  constructor(props) {
    super(props);
  }
  render() {
    const article = this.props.article;
    const nickname = article.who;
    const avatar = "https://"+article.who+"/favicon.ico";
    const image = article.image;
    const caption = article.title + " " + article.snippet;
    const topic = article.topic_text;
    const topic_num = article.topic;
    const doc_id = article.doc_id;
    const date = article.date;
    const topic_link =  'https://'

    return <article className="Post" ref="Post">
      <header>
        <div className="Post-user">
          <div className="Post-user-avatar">
            <a href={article.url}><img src={avatar} alt={nickname} /></a>
          </div>
          <div className="Post-user-nickname">
            <span>{nickname}</span>
          </div>
        </div>
      </header>
      <div className="Post-image">
        <div className="Post-image-bg">
          <img alt={caption} src={image} />
        </div>
      </div>
      <div className='Post-topic'>
        <strong>Topic: </strong> {topic_text}
      </div>
      <div className="Post-caption">
        <strong>{nickname} </strong>{caption}
      </div>
    </article>;
  }
}

export default Post;

