import pandas as pd
import asyncio
from typing import Any, List, Dict
import adalflow as adal
from rate_limiter import async_rate_limited_call, rate_limited_call
import logging
from dataclasses import dataclass, field
import os
import bm25s
import json
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MovieMatch(adal.DataClass):
    """Result of matching a comment to a movie."""

    matched_movies: Dict[str, str] = field(
        metadata={
            "description": """Dictionary where the keys are the imdb_id, title and comment_id.
Example:
{
    "imdb_id": "tt4154756",
    "title": "Avengers: Infinity War",
    "comment_id": "e5puhtm"
}
"""
        }
    )
    confidence_scores: List[float] = field(
        metadata={"description": "Confidence scores for each match"}
    )


class MovieMatcher(adal.Generator):
    """Generator for matching Reddit comments to movies using BM25 retrieval."""

    def __init__(self, model_client=None, model_kwargs=None):
        analysis_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
</output_format>
<examples_of_input_output>
{{example}}
</examples_of_input_output>
<comments>
{{comments}}

"""
        parser = adal.DataClassParser(
            data_class=MovieMatch,
            format_type="json",
            return_data_class=True,
        )

        example = adal.Parameter(
            data="""
<comments>
<comment>Comment ID: e5puhtm Text: Avengers: infinity war? <retrieved_movies><movie>IMDB ID: tt4154756 Title: Avengers: Infinity War by Anthony Russo, Joe Russo</movie> <movie>IMDB ID: tt6331934 Title: Avengers S.T.A.T.I.O.N. by nan</movie> <movie>IMDB ID: tt26736224 Title: Infinity! by Benjamin To</movie></retrieved_movies></comment>\n<comment>Comment ID: e5pvxju Text: The system works! <retrieved_movies><movie>IMDB ID: tt0346994 Title: First Works by nan</movie> <movie>IMDB ID: tt0307023 Title: It works by Fridolin Schönwiese</movie> <movie>IMDB ID: tt0379033 Title: The Works by Gal Katzir</movie></retrieved_movies></comment>\n<comment>Comment ID: e5pw5ev Text: Possibly the animated movie Cars. <retrieved_movies><movie>IMDB ID: tt12291978 Title: The Super Animated Movie by nan</movie> <movie>IMDB ID: tt5396748 Title: Evolution: The Animated Movie by Will Meugniot</movie> <movie>IMDB ID: tt1785581 Title: Storm the Animated Movie by DC Turner</movie></retrieved_movies></comment>\n<comment>Comment ID: e5pz414 Text: That's it. Thanks.  <retrieved_movies><movie>IMDB ID: tt1737768 Title: Thanks by Martin Bergman</movie> <movie>IMDB ID: tt4995112 Title: Thanks by Odeya Rush</movie> <movie>IMDB ID: tt1323960 Title: Thanks by Benedikt Erlingsson</movie></retrieved_movies></comment>\n<comment>Comment ID: e5yzxff Text: Try r/helpmefind  <retrieved_movies><movie>IMDB ID: tt5643266 Title: A Try by nan</movie> <movie>IMDB ID: tt13676666 Title: Behind the Try: A Try Guys Documentary by Jordan Hwang</movie> <movie>IMDB ID: tt1505431 Title: Try Revolution by nan</movie></retrieved_movies></comment>\n<comment>Comment ID: e5yzyic Text: What does he train her to do? <retrieved_movies><movie>IMDB ID: tt0207158 Title: What Does Man Do To Live by Giorgos Papakostas</movie> <movie>IMDB ID: tt0342816 Title: What Made Her Do It? by Shigeyoshi Suzuki</movie> <movie>IMDB ID: tt0150156 Title: What Does My Husband Do at Night? by Michał Waszyński</movie></retrieved_movies></comment>\n<comment>Comment ID: e5z01bp Text: Wait a second.... aren’t u the mod of this place? <retrieved_movies><movie>IMDB ID: tt2938418 Title: QuanTom by Aren Bergstrom</movie> <movie>IMDB ID: tt6942130 Title: LÄCK by Aren Bergstrom</movie> <movie>IMDB ID: tt16918996 Title: Aren't I Reliable? by nan</movie></retrieved_movies></comment>\n<comment>Comment ID: e5z0osh Text: Sure. Thanks. <retrieved_movies><movie>IMDB ID: tt1737768 Title: Thanks by Martin Bergman</movie> <movie>IMDB ID: tt4995112 Title: Thanks by Odeya Rush</movie> <movie>IMDB ID: tt1323960 Title: Thanks by Benedikt Erlingsson</movie></retrieved_movies></comment>\n<comment>Comment ID: e5z1ms6 Text: Yes 😂 <retrieved_movies><movie>IMDB ID: tt2745402 Title: Yes No Yes Yes Go by nan</movie> <movie>IMDB ID: tt12133232 Title: Play or ‘Yes’, ‘Yes’, ‘Yes’ by Barbara Hammer</movie> <movie>IMDB ID: tt0292370 Title: Yes I Said Yes I Will Yes by Phil Solomon</movie></retrieved_movies></comment>\n<comment>Comment ID: e5z1o44 Text: Shoot guns from what I can remember <retrieved_movies><movie>IMDB ID: tt36022388 Title: What I Remember by Alex Hera</movie> <movie>IMDB ID: tt33113532 Title: The Life I Can't Remember by Amy Barrett</movie> <movie>IMDB ID: tt1719571 Title: Paint What You Remember by nan</movie></retrieved_movies></comment>\n<comment>Comment ID: e5z4dzh Text: The professional <retrieved_movies><movie>IMDB ID: tt0082949 Title: The Professional by Georges Lautner</movie> <movie>IMDB ID: tt0339535 Title: The Professional by Dušan Kovačević</movie> <movie>IMDB ID: tt6279058 Title: The Professional by Martín Farina</movie></retrieved_movies></comment>\n<comment>Comment ID: e5z529m Text: Oh, ok, I thought it could’ve been [Leon ](https://www.imdb.com/title/tt0110413/)  but now I’m not sure. \n\nEDIT: actually, yes, I’m very sure.  <retrieved_movies><movie>IMDB ID: tt0220676 Title: Oh, Sure by Richard Condie</movie> <movie>IMDB ID: tt1815764 Title: I Could've Been a Hooker by Baya Kasmi</movie> <movie>IMDB ID: tt0357340 Title: www.anukudumbam.com by O S Gireesh</movie></retrieved_movies></comment>\n<comment>Comment ID: e5z97q1 Text: Yes! Thank you!  <retrieved_movies><movie>IMDB ID: tt3006076 Title: Thank You by V. K. Prakash</movie> <movie>IMDB ID: tt1874532 Title: I Thank You by nan</movie> <movie>IMDB ID: tt1720254 Title: Thank You by Anees Bazmee</movie></retrieved_movies></comment>\n<comment>Comment ID: e5zdsi3 Text: Wonna make me a mod? <retrieved_movies><movie>IMDB ID: tt0023175 Title: Make Me a Star by William Beaudine</movie> <movie>IMDB ID: tt3175476 Title: Make Me Shudder by Poj Arnon</movie> <movie>IMDB ID: tt0048333 Title: Make Me an Offer! by Cyril Frankel</movie></retrieved_movies></comment>\n<comment>Comment ID: e66qpfg Text: Sure soon <retrieved_movies><movie>IMDB ID: tt11730326 Title: Too Soon by H. Kara</movie> <movie>IMDB ID: tt12677092 Title: C U Soon by Mahesh Narayanan</movie> <movie>IMDB ID: tt0238500 Title: Patlachi Soon by nan</movie></retrieved_movies></comment>\n<comment>Comment ID: e66qqbe Text: Finding Nemo 🤔🤔🤔🤔 <retrieved_movies><movie>IMDB ID: tt0266543 Title: Finding Nemo by Andrew Stanton</movie> <movie>IMDB ID: tt6015338 Title: Voices by Gina Nemo</movie> <movie>IMDB ID: tt0387373 Title: Making Nemo by nan</movie></retrieved_movies></comment>\n<comment>Comment ID: e66u5d5 Text: Not sure but I'll take your word for it. Thanks. <retrieved_movies><movie>IMDB ID: tt7962932 Title: I'll Take Your Dead by Chad Archibald</movie> <movie>IMDB ID: tt0193614 Title: I'll Take Your Pain by Mikhail Ptashuk</movie> <movie>IMDB ID: tt0059440 Title: At Midnight I'll Take Your Soul by José Mojica Marins</movie></retrieved_movies></comment>\n<comment>Comment ID: e66u6wa Text: Hey this is a cool post. Hope it takes off. <retrieved_movies><movie>IMDB ID: tt31715619 Title: Harrie Takes Off by nan</movie> <movie>IMDB ID: tt0065769 Title: The Gendarme Takes Off by Jean Girault</movie> <movie>IMDB ID: tt9843292 Title: Fish Takes Off by Deniz Cooper</movie></retrieved_movies></comment>\n<comment>Comment ID: e6pc2vi Text: I found it guys! The last American Virgin, that's the title.  <retrieved_movies><movie>IMDB ID: tt0084234 Title: The Last American Virgin by Boaz Davidson</movie> <movie>IMDB ID: tt0113613 Title: The Last Supper by Stacy Title</movie> <movie>IMDB ID: tt1318044 Title: American Virgin by Clare Kilner</movie></retrieved_movies></comment>\n<comment>Comment ID: e8dbos0 Text: The original War of the Worlds <retrieved_movies><movie>IMDB ID: tt0407304 Title: War of the Worlds by Steven Spielberg</movie> <movie>IMDB ID: tt0046534 Title: The War of the Worlds by Byron Haskin</movie> <movie>IMDB 
ID: tt3154422 Title: War of the Worlds by Cathleen O'Connell</movie></retrieved_movies></comment>
</comments>
Output:
```json
{
    "matched_movies": [
        {
            "imdb_id": "tt4154756",
            "title": "Avengers: Infinity War",
            "comment_id": "e5puhtm"
        }
    ],
    "confidence_scores": [
        0.95
    ]
}
""",
            param_type=adal.ParameterType.DEMOS,
            requires_opt=True,
            role_desc="Example of how to match a comment to a movie.",
        )

        prompt_params = {
            "example": example,
            "output_format_str": parser.get_output_format_str(),
            "task_desc": adal.Parameter(
                data="""You are a movie matching expert. Your task is to analyze a Reddit comment and match any suggested movie titles to their corresponding IMDB IDs.

INSTRUCTIONS:
1. ANALYZE the comments carefully in the context of the post title and text question that is provided to identify any movie titles mentioned, some comments do not mention any movie titles so just ignore them.
2. For each identified comment that mentions a movie title, provide the imdb_id, title and comment_id if the comment doesn't mention any movie titles, just ignore it.
3. For each match:
   - Extract the suggested title from the comment
   - Match it with the most relevant movie from the provided matches or provide imdb id and title from your memory.
   - Assign a confidence score (0-1) based on how well the titles match
   - Include the IMDB ID in the result.

Your analysis should be thorough and accurate, ensuring that movie titles are properly matched to their IMDB IDs.
""",
                param_type=adal.ParameterType.PROMPT,
                requires_opt=True,
            ),
        }

        super().__init__(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=analysis_template,
            prompt_kwargs=prompt_params,
            output_processors=parser,
        )


def load_dataset(filename: str) -> pd.DataFrame:
    """Load and preprocess dataset based on filename."""
    if filename == "TMDB_all_movies.csv":
        df = pd.read_csv(filename)
        # Select and rename required columns
        df = df[["title", "director", "imdb_id"]].copy()
        df.dropna(subset=["imdb_id"], inplace=True)
        # Create combined title-director string
        df["title_director"] = df.apply(
            lambda x: f"{x['title']} by {x['director']}", axis=1
        )
        return df
    else:
        raise ValueError(f"Unsupported dataset: {filename}")


def movie_metadata_to_docs(movie_data: pd.DataFrame) -> List[adal.Document]:
    """Convert movie metadata to a list of strings."""
    return [
        adal.Document(text=row["title"], meta_data={"imdb_id": row["imdb_id"]})
        for _, row in movie_data.iterrows()
    ]


def context2string(
    comment_id: str,
    comment_text: str,
    retriever_output: adal.RetrieverOutput,
    docs: List[adal.Document],
) -> str:
    """Convert a comment to a string."""
    return f"<comment> Comment ID: {comment_id} Text: {comment_text} {retriever_output2string(retriever_output, docs)}</comment>"


def context2string(
    post_text: str,
    comment_id: str,
    comment_text: str,
    movies_ids: List[str],
    movies_texts: List[str],
) -> str:
    """Convert a comment in context to a string."""
    movie_strings = [
        f"<movie>IMDB ID: {movie_id} Title: {movie_text}</movie>"
        for movie_id, movie_text in zip(movies_ids, movies_texts)
    ]
    return f"<comment> {post_text} Comment ID: {comment_id} Text: {comment_text} <retrieved_movies>{' '.join(movie_strings)}</retrieved_movies></comment>"


def retriever_output2string(
    retriever_output: adal.RetrieverOutput, docs: List[adal.Document]
) -> str:
    """Convert a retriever output to a string."""
    doc_strings = [
        f"<movie>IMDB ID: {docs[doc_idx].meta_data['imdb_id']} Title: {docs[doc_idx].text}</movie>"
        for doc_idx in retriever_output.doc_indices
    ]
    return f"<retrieved_movies>{''.join(doc_strings)}</retrieved_movies>"


def get_comment_post_text(
    comment_id: str, posts: pd.DataFrame, posts_by_comment_id: Dict[str, Any]
) -> str:
    post_row = posts[posts['post_id'] == posts_by_comment_id[comment_id]]
    if len(post_row) == 0:
        return ""
    return f"Post Title: {post_row['title'].values[0]} Post Text: {post_row['text'].values[0]}"


async def process_reddit_comments(
    *,
    posts_data: pd.DataFrame,
    comments_data: pd.DataFrame,
    comments_in_context: pd.DataFrame,
    model_client: adal.ModelClient,
    model_kwargs: Dict[str, Any],
    output_file: str = "movie_matches.jsonl",
    batch_size: int = 20,
):
    matcher = MovieMatcher(
        model_client=model_client,
        model_kwargs=model_kwargs,
    )

    comments_in_context = (
        comments_in_context.groupby(["comment_id", "comment_text"])
        .agg({"movie_id": list, "movie_title": list})
        .reset_index()
    )
    comment_ids = comments_in_context["comment_id"].tolist()
    posts_by_comment_id = posts_data.loc[posts_data["post_id"].isin(comment_ids)][
        ["post_id", "comment_id"]
    ].set_index("comment_id").to_dict(orient="records")
    comments_in_context = comments_in_context.to_dict(orient="records")
    comments_in_context = [
        context2string(
            get_comment_post_text(comment["comment_id"], posts_by_comment_id),
            comment["comment_id"],
            comment["comment_text"],
            comment["movie_id"],
            comment["movie_title"],
        )
        for comment in comments_in_context
    ]

    messages = [
        "\n".join(comments_in_context[i : i + batch_size])
        for i in range(0, len(comments_in_context), batch_size)
    ]

    for i in range(0, len(messages), 5):
        batch = messages[i : i + 5]
        results = await asyncio.gather(
            *[process_comment_batch(matcher, message) for message in batch]
        )
        for result in results:
            if result is None:
                continue
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()))
                f.write("\n")


def sync_process_reddit_comments(
    *,
    posts_data: pd.DataFrame,
    comments_data: pd.DataFrame,
    comments_in_context: pd.DataFrame,
    model_client: adal.ModelClient,
    model_kwargs: Dict[str, Any],
    output_file: str = "movie_matches.jsonl",
    batch_size: int = 20,
):
    matcher = MovieMatcher(
        model_client=model_client,
        model_kwargs=model_kwargs,
    )

    comments_in_context = (
        comments_in_context.groupby(["comment_id", "comment_text"])
        .agg({"movie_id": list, "movie_title": list})
        .reset_index()
    )
    comments_in_context = comments_in_context.merge(comments_data[["comment_id", "post_id"]], on="comment_id", how="left")
    comments_in_context = comments_in_context.merge(posts_data[["post_id", "title", "text"]], on="post_id", how="left")
    comments_in_context = comments_in_context.drop_duplicates(subset=["comment_id"])

    comments_in_context = comments_in_context.to_dict(orient="records")
    comments_in_context = [
        context2string(
            f"Post Title: {comment['title']} Post Text: {comment['text']}",
            comment["comment_id"],
            comment["comment_text"],
            comment["movie_id"],
            comment["movie_title"],
        )
        for comment in comments_in_context
    ]

    messages = [
        "\n".join(comments_in_context[i : i + batch_size])
        for i in range(0, len(comments_in_context), batch_size)
    ]

    for comment in messages:
        try: 
            result = rate_limited_call("google", matcher.call, prompt_kwargs={"comments": comment})
        except Exception as e:
            logger.error(f"Error processing comment: {e}")
            continue
        if result is None:
            continue
        if result.data is None:
            continue
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.data.to_dict()))
            f.write("\n")


async def process_comment_batch(matcher: MovieMatcher, comments: str):
    """Process a single comment and match it with movies."""

    # Generate structured response
    result = await async_rate_limited_call(
        "google", matcher.acall, prompt_kwargs={"comments": comments}
    )

    return result.data


def preprocess_comments(
    comments_data: pd.DataFrame, movie_data: pd.DataFrame, batch_size: int = 50
) -> pd.DataFrame:
    """Preprocess comments data."""
    movie_data_corpus = movie_data["title_director"].to_list()
    movie_data_ids = movie_data["imdb_id"].to_list()
    logger.info("Building index")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(movie_data_corpus))
    logger.info("Index built")
    data_rows = []
    comments_ids = comments_data["comment_id"].tolist()
    comment_texts = comments_data["body"].tolist()
    ids, scores = retriever.retrieve(bm25s.tokenize(comment_texts), k=3)
    ids = ids.tolist()
    scores = scores.tolist()
    for comment_id, comment_text, doc_ids, scores in zip(
        comments_ids, comment_texts, ids, scores
    ):
        for doc_id, score in zip(doc_ids, scores):
            data_row = {
                "comment_id": comment_id,
                "comment_text": comment_text,
                "movie_id": movie_data_ids[doc_id],
                "movie_title": movie_data_corpus[doc_id],
                "score": score,
            }
            data_rows.append(data_row)
    return data_rows


def preprocess_comments_and_save():
    comments_data = pd.read_csv("movies_comments.csv")
    movie_data = load_dataset("TMDB_all_movies.csv")

    comments_in_context = preprocess_comments(comments_data, movie_data)
    with open("comments_in_context.jsonl", "w", encoding="utf-8") as f:
        for comment in comments_in_context:
            f.write(json.dumps(comment))
            f.write("\n")


def main():
    if os.path.exists("comments_in_context.jsonl"):
        with open("comments_in_context.jsonl", "r", encoding="utf-8") as f:
            comments_in_context = pd.read_json("comments_in_context.jsonl", lines=True)
    else:
        preprocess_comments_and_save()
        comments_in_context = pd.read_json("comments_in_context.jsonl", lines=True)

    comments_data = pd.read_csv("movies_comments.csv")
    posts_data = pd.read_csv("movies_posts.csv")

    model_client = adal.GoogleGenAIClient()
    model_kwargs = {
        "model": "gemini-2.0-flash",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
    }
    sync_process_reddit_comments(
        comments_data=comments_data,
        comments_in_context=comments_in_context,
        posts_data=posts_data,
        model_client=model_client,
        model_kwargs=model_kwargs,
        output_file="movie_matches_gemini.jsonl",
    )

    # model_client = adal.OpenAIClient(base_url="https://openrouter.ai/api/v1")
    # model_kwargs = {
    #     "model": "qwen/qwq-32b:free",
    #     "temperature": 0.6,
    #     "top_p": 0.95,
    # }

    # Process comments
    # await process_reddit_comments(
    #     comments_data=comments_data,
    #     comments_in_context=comments_in_context,
    #     posts_data=posts_data,
    #     model_client=model_client,
    #     model_kwargs=model_kwargs,
    #     output_file="movie_matches_gemini.jsonl",
    # )


if __name__ == "__main__":
    # asyncio.run(main())
    main()
