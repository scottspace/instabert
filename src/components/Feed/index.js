// src/components/contacts.js
import React from 'react';
import Post from '../Post';

const Feed = ({ articles }) => {
  return (
    <div>
      {articles.map((article) => (
        <Post article={article}  />
      ))}
    </div>
  )
};

export default Feed;
