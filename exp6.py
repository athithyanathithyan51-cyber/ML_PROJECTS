
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import implicit
import warnings
warnings.filterwarnings("ignore")

books = pd.read_csv("Books.csv", encoding="latin-1", on_bad_lines="skip")
ratings = pd.read_csv("Ratings.csv", encoding="latin-1", on_bad_lines="skip")

ratings = ratings[ratings["Book-Rating"] > 0]
books = books.drop_duplicates("ISBN")

users = sorted(ratings["User-ID"].unique())
items = sorted(ratings["ISBN"].unique())

user_map = {u: i for i, u in enumerate(users)}
book_map = {b: i for i, b in enumerate(items)}

rows = ratings["User-ID"].map(user_map)
cols = ratings["ISBN"].map(book_map)

matrix = coo_matrix(
    (ratings["Book-Rating"], (rows, cols)),
    shape=(len(users), len(items))
).tocsr()

model = implicit.als.AlternatingLeastSquares(
    factors=50, regularization=0.1, iterations=20
)
model.fit(matrix)



def recommend(user_id):
    if user_id not in user_map:
        return books[["Book-Title", "Book-Author"]].head(5)

    uid = user_map[user_id]
    user_row = matrix[uid]

    ids, scores = model.recommend(
        userid=uid,
        user_items=user_row,
        N=5,
        filter_already_liked_items=True
    )

    recommended_isbns = [items[i] for i in ids]
    result = books[books["ISBN"].isin(recommended_isbns)]

    return result[["Book-Title", "Book-Author"]].drop_duplicates().head(5)


for user in users[:5]:
    print("\nRecommendations for User", user)
    print(recommend(user).to_string(index=False))
