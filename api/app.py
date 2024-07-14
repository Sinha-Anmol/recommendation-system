from flask import Flask, request, jsonify
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Connect to the database
engine = create_engine('mysql+pymysql://root:@localhost/recommendation_db')

# Load user behavior data
df = pd.read_sql('SELECT * FROM user_behavior', con=engine)

# Convert interaction types to numerical values
interaction_mapping = {'view': 1, 'like': 2, 'comment': 3}
df['interaction_value'] = df['interaction_type'].map(interaction_mapping)

# Pivot the data to create a user-content interaction matrix
interaction_matrix = df.pivot_table(index='user_id', columns='content_id', values='interaction_value', fill_value=0)

# Compute the cosine similarity between users
user_similarity = cosine_similarity(interaction_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)

# Compute the cosine similarity between content
content_similarity = cosine_similarity(interaction_matrix.T)
content_similarity_df = pd.DataFrame(content_similarity, index=interaction_matrix.columns, columns=interaction_matrix.columns)

def recommend_content(user_id, num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    recommended_content = set()
    for similar_user in similar_users:
        user_content = df[df['user_id'] == similar_user]['content_id'].values
        recommended_content.update(user_content)
        if len(recommended_content) >= num_recommendations:
            break
    
    # Adding content-based filtering
    user_interacted_content = df[df['user_id'] == user_id]['content_id'].values
    for content_id in user_interacted_content:
        similar_content = content_similarity_df[content_id].sort_values(ascending=False).index[1:num_recommendations]
        recommended_content.update(similar_content)
        if len(recommended_content) >= num_recommendations:
            break
    
    return [int(content_id) for content_id in list(recommended_content)[:num_recommendations]]

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    recommendations = recommend_content(user_id)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
