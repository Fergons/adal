import functools
import os
import opik.opik_context
import orjson
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import pandas as pd

import adalflow as adal
from adalflow.core.types import DataClass, Document
from adalflow.components.retriever.faiss_retriever import FAISSRetriever
from m3_embedder import M3Embedder
from rate_limiter import rate_limited_call
import asyncio
import opik

import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TropeSimilarityCategory(str, Enum):
    """Categories of tropes based on their utility for story similarity assessment."""

    PLOT_STRUCTURE = "plot_structure"  # Overall plot patterns (e.g., Hero's Journey, Cinderella plot)
    CHARACTER_ARCHETYPE = "character_archetype"  # Character types and roles (e.g., Chosen One, Wise Mentor)
    RELATIONSHIP_DYNAMIC = (
        "relationship_dynamic"  # Character relationships (e.g., Enemies to Lovers)
    )
    THEME = "theme"  # Thematic elements (e.g., Coming of Age, Redemption)
    SETTING = (
        "setting"  # Setting-specific tropes (e.g., Boarding School, Post-Apocalyptic)
    )
    NARRATIVE_DEVICE = (
        "narrative_device"  # Storytelling techniques (e.g., Unreliable Narrator)
    )
    GENRE_SPECIFIC = (
        "genre_specific"  # Tropes specific to genres (e.g., Final Girl in horror)
    )
    MINOR_ELEMENT = (
        "minor_element"  # Small details unlikely to indicate story similarity
    )
    NOT_RELEVANT = "not_relevant"  # Not useful for story similarity assessment


@dataclass
class TropeAnalysisResult(DataClass):
    """Result of analyzing a trope for story similarity assessment."""

    similarity_relevance_score: int = field(
        default=0,
        metadata={
            "description": "Score from 1-5 indicating how useful this trope is for assessing story similarity (5 being highest)."
        },
    )

    primary_category: TropeSimilarityCategory = field(
        default=TropeSimilarityCategory.NOT_RELEVANT,
        metadata={
            "description": "Primary category this trope falls into for similarity assessment."
        },
    )

    secondary_categories: List[TropeSimilarityCategory] = field(
        default_factory=list,
        metadata={"description": "Secondary categories this trope may fall into."},
    )

    reasoning: str = field(
        default_factory=str,
        metadata={
            "description": "Detailed reasoning for the categorization and relevance score."
        },
    )

    example_similar_stories: List[str] = field(
        default_factory=list,
        metadata={
            "description": "Example of books or movies that would be considered similar based on this trope."
        },
    )

    keywords: List[str] = field(
        default_factory=list,
        metadata={
            "description": "Key terms that could be used to identify this trope in text."
        },
    )
    __input_fields__ = []
    __output_fields__ = [
        "similarity_relevance_score",
        "primary_category",
        "secondary_categories",
        "reasoning",
        "example_similar_stories",
        "keywords",
    ]


class TropeAnalyzer(adal.Generator):
    """Generator for analyzing tropes for story similarity assessment."""

    def __init__(self, model_client=None, model_kwargs=None):
        analysis_template = r"""
{{task_desc}}
<output_format>
{{output_format_str}}
</output_format>
<trope>
Name: {{trope_name}}
Description: {{trope_description}}
</trope>
"""
        parser = adal.DataClassParser(
            data_class=TropeAnalysisResult,
            format_type="json",
            return_data_class=True,
        )
        prompt_params = {
            "output_format_str": parser.get_output_format_str(),
            "task_desc": adal.Parameter(
                data="""You are a literary analysis expert specializing in narrative tropes and comparative literature. Your task is to analyze a trope and assess its relevance for determining story similarity between different works.

INSTRUCTIONS:
1. ANALYZE the trope name and description carefully.
2. EVALUATE how useful this trope would be for determining if two stories are similar.
3. ASSIGN a similarity relevance score from 1-10:
   - 5: Trope that can be used to find very similar stories. e.g. Retellings, sequels, prequels, adaptations, etc.
   - 4: Somewhat useful trope and specific to a particular story that is strong indicator of story similarity.
   - 3: Notable narrative element that contributes to but doesn't define story identity. e.g. Character traits, plot twists, etc.
   - 2: Minor element that adds flavor but doesn't significantly impact story similarity. e.g. Character names, minor plot points, etc.
   - 1: Not useful for story similarity assessment. Some generic trope that is not specific to a particular story but to a large genre.

4. CATEGORIZE the trope into its primary category from these options:
   - plot_structure: Overall plot patterns (e.g., Hero's Journey, Cinderella plot)
   - character_archetype: Character types and roles (e.g., Chosen One, Wise Mentor)
   - relationship_dynamic: Character relationships (e.g., Enemies to Lovers)
   - theme: Thematic elements (e.g., Coming of Age, Redemption)
   - setting: Setting-specific tropes (e.g., Boarding School, Post-Apocalyptic)
   - narrative_device: Storytelling techniques (e.g., Unreliable Narrator)
   - genre_specific: Tropes specific to genres (e.g., Final Girl in horror)
   - minor_element: Small details unlikely to indicate story similarity
   - not_relevant: Not useful for story similarity assessment

5. IDENTIFY any secondary categories the trope might also belong to.

6. PROVIDE detailed reasoning for your categorization and relevance score.

7. LIST examples of stories that would be considered similar based on this trope.

8. SUGGEST keywords that could be used to identify this trope in text.

EVALUATION CRITERIA:
- SPECIFICITY: How specific is the trope to a particular type of story?
- CENTRALITY: How central is the trope to a story's identity?
- DISTINCTIVENESS: How distinctive is the trope in differentiating stories?
- RECOGNIZABILITY: How easily recognizable is the trope across different works?
- CONSISTENCY: How consistently is the trope applied across different works?

Your analysis should be thorough, nuanced, and focused on the trope's utility for comparing story similarity.
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


async def analyze_tropes(
    tropes_data: pd.DataFrame,
    model_client,
    model_kwargs,
    output_file="trope_analysis_results.jsonl",
    batch_size: int = 4,
):
    gen = TropeAnalyzer(
        model_client=model_client,
        model_kwargs=model_kwargs,
    )

    
    partial = functools.partial(rate_limited_call, "openrouter", analyze_trope, gen)
    for i in range(0, len(tropes_data), batch_size):
        # create batch of 8 rows
        batch = tropes_data.iloc[i : i + batch_size]
        results = await asyncio.gather(
            *[
                partial(row["trope_name"], row["trope_description"])
                for _, row in batch.iterrows()
            ]
        )
        for result, (_, row) in zip(results, batch.iterrows()):
            with open(output_file, "ab") as f:
                try:
                    f.write(
                        orjson.dumps(
                            result.to_dict().update({"trope_name": row["trope_name"]})
                        )
                    )
                    f.write(b"\n")
                except Exception as e:
                    logger.error(
                        f"Error writing result for trope {row['trope_name']}: {e}"
                    )
    return results


@opik.track(project_name="tvtrope-analysis")
async def analyze_trope(gen: TropeAnalyzer, trope_name: str, trope_description: str):
    """
    Analyze a list of tropes for story similarity assessment.

    Args:
        tropes_data: DataFrame or list of dictionaries containing trope information
        model_client: The LLM client to use for analysis
        model_kwargs: Model configuration parameters
        output_file: Path to save analysis results

    Returns:
        List of TropeAnalysisResult objects
    """
    result = await gen.acall(
        prompt_kwargs={"trope_name": trope_name, "trope_description": trope_description}
    )
    return result.data


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze tropes for story similarity assessment"
    )
    parser.add_argument(
        "--input", type=str, required=False, help="Path to JSON file containing tropes"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trope_analysis_results.jsonl",
        help="Path to save analysis results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model to use for analysis",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for model generation",
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize model client
    # model_client = adal.GoogleGenAIClient()
    # model_kwargs = {
    #     "model": args.model,
    #     "temperature": args.temperature,
    #     "top_p": 0.95,
    #     "top_k": 40,
    #     "max_output_tokens": 8192,
    #     "response_mime_type": "text/plain",
    # }
    
    model_client = adal.OpenAIClient(base_url="https://openrouter.ai/api/v1")
    model_kwargs = {
        "model": "qwen/qwq-32b:free",
        "temperature": 1.0,
        "top_p": 1.0,
    }

    if args.input:
        with open(args.input, "r", encoding="utf-8") as fi:
            tropes = pd.read_csv(fi)
            tropes.rename(
                columns={
                    "Trope": "trope_name",
                    "Description": "trope_description",
                    "TropeID": "trope_id",
                },
                inplace=True,
            )
            tropes.drop_duplicates(subset=["trope_description"], inplace=True)
            tropes.dropna(inplace=True)
            tropes = filter_trope_experiment_by_already_analyzed(tropes)

    else:
        tropes = pd.DataFrame(
            {
                "trope_id": [0],
                "trope_name": ["GeneralFailure"],
                "trope_description": [
                    """
They are utterly ruthless, unfettered, and fanatically dedicated to the destruction of their enemies. Whether fighting for good or evil, they have no qualms with employing the cruelest, foulest, most abominable strategems and minions — using every means both fair and foul in the pursuit of their goals. Their limitless ambition and cunning make them the very epitome of martial brilliance...
...or at least they would if they weren't also a gibbering moron who puts their soldiers at risk.
For some reason, villainous organizations which have no problem with kidnapping, blackmailing, threatening the destruction of the world, or even kicking puppies, somehow tolerate having an idiotic leader whose inept schemes for world domination are always foiled, often because of the utterly bizarre plans and implementation that General Failure himself is responsible for. Oh, they might bitch and moan about the dumb ideas, but it's not like they'll ever do anything about it.
Occasionally, The Watson, the Meta Guy, or other characters will question the Big Bad's ludicrous schemes, but since they're not in charge, that will be it. Very often The Starscream is the only one who opposes the leader at all, making him look like the Only Sane Man on their side.
Common phrases of the General usually include "I wrote the manual on military tactic X," or "This reminds me of the time we fought enemy X in an improbable location, what a tale that is!"
General Failure may have started out as a competent commander in a position of less importance, and his success led to him being promoted beyond his capabilities. If this is the case, then it's a villainous example of The Peter Principle. If he started out as an incompetent mook or private, you can expect his rise to be an improbable series of Kicked Upstairs, Uriah Gambit, and Promoted to Scapegoat that never deliver on the bad ending or being the only living replacement left when his superiors keep dying.
Most of the time the leader is also a Bad Boss, which can lead to We Have Reserves and possibly Mook Depletion. One wonders sometimes if the good guys are secretly making sure the doofus on top stays there. General Failure is essentially the personification of Failure Is the Only Option, is the classic example of being Lethally Stupid, and is the eventual destination of severe Villain Decay. He often bears similarities to The Neidermeyer, but on a much higher scale. Compare Armchair Military, Miles Gloriosus, Modern Major General, Lord Error-Prone. Pointy-Haired Boss is a similar non-military trope.
Contrast Four-Star Badass, General Ripper, Colonel Badass, Sergeant Rock, and Surrounded by Idiots.
This trope does not happen too much in Real Life. Really incompetent officers usually never even graduate from the military academy: incompetent officers mostly don't tend to get promoted past Captain (Lieutenant in the Navy) level. Most real-life officers appearing as General Failures are simply unlucky ones (and conversely, many military "geniuses" just got lucky and afterwards announced I Meant to Do That).
Granted, in the mid-19th Century and prior, there was (more room for) nepotism, and military ranks and jobs had to be bought and were only available to people of the right class/social standing. But even then, there were limits to how much incompetence a military establishment would tolerate before either you got demoted, or some of the people dying under your command saw to it you got hit by a stray bullet, or you and your remaining troops were captured by a foe with more competent leaders.
This isn't to say incompetence doesn't exist within Real Life militaries, of course. It's just that usually, it can't be pinned down to one person. More often than not, bad decisions are made by several people due to a combination of factors involving miscommunication between personnel, an outdated and hopelessly bloated Vast Bureaucracy, a culture and power structure that encourages higher ranking individuals to interfere in the work of lower ranking personnel despite having no expertise in their field (think Executive Meddling, but for armies instead of TV Networks), and to top it all off, good old groupthink. In other words, it's not a leadership problem, it's a structural problem.
Real-life General Failures may have existed and were quite notorious, but they alone were only responsible for a fraction of military blunders throughout history.
Still, No Real Life Examples, Please!
"""
                ],
            }
        )

    await analyze_tropes(tropes, model_client, model_kwargs, args.output)


def filter_trope_experiment_by_already_analyzed(tropes: pd.DataFrame = None):
    client = opik.Opik(project_name="tvtrope-analysis")
    traces = client.search_traces(project_name="tvtrope-analysis", filter_string='input contains "qwen"', max_results=10000)
    tropes_already_analyzed = [trace.input["trope_name"] for trace in traces]
    logger.info(f"Found {len(tropes_already_analyzed)} tropes already analyzed")
    df = tropes[~tropes["trope_name"].isin(tropes_already_analyzed)]
    logger.info(f"Need to analyze {len(df)} tropes from {len(tropes)}.")
    return df


if __name__ == "__main__":
    asyncio.run(main())
