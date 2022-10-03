import io
import json

import requests
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from streamlit_agraph import agraph, Node, Edge, Config

from src import keyselect
from src.utils import tweet_parser

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

if "labelled_count" not in st.session_state:
    st.session_state.labelled_count = 0

if "next_keyword" not in st.session_state:
    st.session_state.next_keyword = None

make_tweet_url = lambda tid: f"https://twitter.com/any/status/{tid}"


class Tweet(object):
    """Embed tweet in app while labelling keywords"""

    def __init__(self, s, embed_str=False):
        self.url = s
        if not embed_str:
            api = "https://publish.twitter.com/oembed?url={}&theme=dark".format(s)
            response = requests.get(api)
            self.text = response.json()["html"]
        else:
            self.text = s

    def _repr_html_(self):
        return self.text

    def component(self):
        return components.html(self.text, height=450, scrolling=True)


##### defining sidebar #####
st.sidebar.markdown(
    "<h1 style=''>DisKeyword 🧭</h1>",
    unsafe_allow_html=True,
)
st.sidebar.header("Settings")
topic = st.sidebar.text_input(
    "Name your topic:",
    placeholder="topic of study",
    help="The chosen topic will be used to in saved configuration files.",
)

tweet_file = st.sidebar.file_uploader(
    "Upload Twitter data:",
    type=["json"],
    help="Upload a JSON file containing a list of tweets as collected using tweepy. Both version 3.10.0 and 4.0.0 are supported. Note that earlier (and later) versions may be too, however they haven't been tested for. The maximum file size is 200MB.",
)
if tweet_file is not None and "keyselect" not in st.session_state:
    raw_tweets = json.load(tweet_file)
    # extract hashtags relations from tweets
    relation_parser = tweet_parser.StandardRelationParser()
    hashtag_relations, ind_days = [], set()
    tweet_metadata = {}
    for raw_tweet in raw_tweets:
        tweet_metadata[raw_tweet.get("id_str")] = relation_parser.parse_metadata(
            raw_tweet
        )
        tmp_relations, tmp_days = relation_parser.parse_relations_from_tweet(raw_tweet)
        hashtag_relations.extend(tmp_relations.get("hashtags"))
        ind_days.update(set(tmp_days))

    # redefine days as strings, sorted in increasing order
    ind_days = [day.strftime("%Y-%m-%d") for day in sorted(list(ind_days))]
    # construct tweet-hashtag dictionnary
    keyselect_obj = keyselect.KeySelect(
        hashtag_relations,
        ind_days,
        tweet_metadata=tweet_metadata,
        baseline_keywords=None
        if "seed_keywords" not in st.session_state
        else set(st.session_state["seed_keywords"]),
    )
    st.session_state["keyselect"] = keyselect_obj

date_slider = None

if (
    "updated_keyselect_budget" in st.session_state
    and "keyselect" in st.session_state
    and st.session_state.updated_keyselect_budget
):
    st.session_state[
        "keyselect"
    ].remaining_budget = st.session_state.updated_keyselect_budget
    st.session_state.updated_keyselect_budget = None

keyword_file = st.sidebar.file_uploader(
    "Upload keyword file:",
    type=["json", "txt"],
    help="Upload your keywords as a text (one keyword per line) or JSON (contained in a list) file. The file size must remain below 200MB.",
)
if keyword_file is not None and "seed_keywords" not in st.session_state:
    parsed_keyword_file = io.StringIO(keyword_file.getvalue().decode("utf-8"))
    parsed_keywords = json.loads(parsed_keyword_file.read())
    st.session_state["seed_keywords"] = parsed_keywords
    # update the baseline keywords if object there
    if "keyselect" in st.session_state:
        st.session_state.keyselect.set_baseline_keywords(set(parsed_keywords))

labelling_budget = st.sidebar.select_slider(
    "Keyword labelling budget",
    [7, 12, 30, 100],
    30,
    help="Determine the maximum number of keywords to consider on a daily basis before switching to the next day of data.",
)
##### end of sidebar #####

with st.expander("Quick Start"):
    st.markdown(
        "\n Find keywords relevant to your topic of study directly from your Twitter data. Use the sidebar on the left to **upload your tweets and keywords**, as well as set a name for your topic. Once your data has been uploaded, **label your tweets below**. Every time you mark a tweet as being relevant to your topic, it automatically gets added to the list of keywords displayed in the `Labelled Topic Keywords` section below. Once you're done with your keyword selection, download them by clicking the `Download Keywords` button.\n\n If you have any comments or find a bug, feel free to report an issue on the [KeySelect GitHub repository](https://github.com/sachalevy/active-keyword-selection), send an email to sacha.levy@mail.mcgill.ca, or a Twitter DM to @sachalevy3.\n",
    )


def reset_keyword_multiselect():
    global new_positive_keywords, new_negative_keywords
    new_positive_keywords = list()
    new_negative_keywords = list()


with st.expander("Labelled Keywords"):
    # check whether some keywords have been uploaded
    if "seed_keywords" not in st.session_state:
        st.info("Upload keywords in sidebar")
    elif "keyselect" not in st.session_state:
        st.info("Upload dataset in sidebar")
    else:
        col1, col2 = st.columns(2)
        new_positive_keywords = col1.multiselect(
            "Label new positive keywords",
            options=list(st.session_state.keyselect.hashtag_idx),
            help="Label new positive keywords from hashtags found in tweet dataset.",
        )
        st.session_state.seed_keywords = list(
            set(st.session_state.seed_keywords).union(set(new_positive_keywords))
        )
        new_negative_keywords = col1.multiselect(
            "Label a new negative keywords",
            options=list(st.session_state.keyselect.hashtag_idx),
            help="Label new negative keywords from hashtags found in tweet dataset.",
        )
        st.session_state.keyselect.negative_keywords.update(set(new_negative_keywords))

        json_keywords_display = col2.empty()
        json_keywords_display.json(
            {
                "positive": st.session_state.seed_keywords,
                "negative": list(st.session_state.keyselect.negative_keywords),
            },
        )
        download_current_keywords = col2.download_button(
            "Download keywords",
            " ".join(st.session_state.seed_keywords),
            help="Download the current list of labelled keywords.",
        )


# check whether tweets have been uploaded
with st.expander("Dataset Overview"):
    if "keyselect" not in st.session_state:
        st.info("Upload dataset in sidebar")
    elif "seed_keywords" not in st.session_state:
        st.info("Upload keywords in sidebar")
    else:
        col1, col2 = st.columns(2)
        col1.metric(
            label="Significant day span",
            value=len(st.session_state.keyselect.ind_days),
            help="Number of days in the dataset counting 100 or more tweets.",
        )
        col2.metric(
            label="Total tweet-hashtag relation count",
            value=f"{st.session_state.keyselect.relation_count:,}",
            help="Total number of tweet-hashtag relations observed over significant days in the dataset.",
        )

        # build hashtag network dicts for visualization
        th_dict, ht_dict = {}, {}
        for day in st.session_state.keyselect.day_batches:
            for pair in st.session_state.keyselect.day_batches[day]:
                tweet, hashtag = (
                    st.session_state.keyselect.idx_tweet[pair[1]],
                    st.session_state.keyselect.idx_hashtag[pair[0]],
                )
                if tweet not in th_dict:
                    th_dict[tweet] = {}

                if hashtag not in th_dict[tweet]:
                    th_dict[tweet][hashtag] = 0
                th_dict[tweet][hashtag] += 1

                if hashtag not in ht_dict:
                    ht_dict[hashtag] = {}

                if tweet not in ht_dict[hashtag]:
                    ht_dict[hashtag][tweet] = 0
                ht_dict[hashtag][tweet] += 1

        tab1, tab2 = st.tabs(["Tweets", "Hashtags"])
        with tab1:
            col1a, col2a, col3a = st.columns(3)
            col1a.metric(
                label="Tweet count",
                value=f"{st.session_state.keyselect.tweet_count:,}",
                delta=f"{st.session_state.keyselect.tweet_count-sum([st.session_state.keyselect.daily_tweet_count[d] for d in st.session_state.keyselect.daily_tweet_count]):,}",
                help="Total number of tweets observed on significant days (red number indicates difference when all days are included).",
            )
            sorted_days = sorted(st.session_state.keyselect.daily_tweet_count.keys())
            col3a.metric(
                label="Total day span",
                value=len(sorted_days),
                delta=len(st.session_state.keyselect.ind_days) - len(sorted_days),
                help="The total day span is the number of days in the dataset. The significant day span is the number of days with more than 100 tweets.",
            )
            col2a.metric(
                label="User count",
                value=f"{st.session_state.keyselect.user_count:,}",
                help="Total number of tweeting users observed on significant days.",
            )

            col1, col2 = st.columns(2)

            fig = px.line(
                x=sorted(st.session_state.keyselect.ind_days),
                y=[
                    st.session_state.keyselect.daily_tweet_count[day]
                    for day in sorted(st.session_state.keyselect.ind_days)
                ],
                title="Daily tweet count",
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Tweet count",
            )
            col1.plotly_chart(
                fig,
                use_container_width=True,
                help="Displays the tweet count for every day with more than 100 tweets in the dataset.",
            )

            observed_day = col2.selectbox(
                label="Select a day to visualize its hashtags trends",
                options=list(st.session_state.keyselect.ind_days),
                index=0,
                help="Select a day to visualize its hashtag trends.",
            )

            # generate a summary for all hashtags on that day
            day_text = ""
            day_idx = st.session_state.keyselect.day_idx_dict.get(observed_day)
            for hashtag in st.session_state.keyselect.daily_hashtag_occurences[day_idx]:
                for _ in range(
                    st.session_state.keyselect.daily_hashtag_occurences[day_idx][
                        hashtag
                    ]
                ):
                    day_text += f"{hashtag} "

            wordcloud = WordCloud(
                max_words=64,
                background_color="rgba(15,17,22,0)",
                width=3000,
                height=2000,
                colormap="plasma",
                collocations=False,
            ).generate(day_text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.set_axis_off()
            plt.tight_layout(pad=0)
            fig.set_facecolor((0.059, 0.067, 0.086))
            col2.pyplot(fig)

        with tab2:
            col1, col2 = st.columns(2)
            col1.metric(
                label="Hashtag count",
                value=f"{st.session_state.keyselect.hashtag_count:,}",
                help="Total number of hashtags observed on significant days in the dataset.",
            )

            hashtag_occurences = [len(th_dict[tweet]) for tweet in th_dict]
            col1.metric(
                label="Median hashtag count per tweet",
                value=f"{np.median(hashtag_occurences):.2f}",
                help="Median of the hashtag count per tweet over significant days of the dataset.",
            )
            col1.metric(
                label="Mean hashtag count per tweet",
                value=f"{np.mean(hashtag_occurences):.2f}",
                help="Mean of the hashtag count per tweet over significant days of the dataset.",
            )
            col1.metric(
                label="Std hashtag count per tweet",
                value=f"{np.std(hashtag_occurences):.2f}",
                help="Standard deviation of the hashtag count per tweet over significant days of the dataset.",
            )

            observed_hashtag = col2.selectbox(
                label="Select a hashtag to explore its neighbourhood",
                options=list(st.session_state.keyselect.hashtag_idx.keys()),
                index=0,
                help="Select a hashtag to explore its surrounding nodes in the flattened tweet-hashtag network over the whole dataset. Hashtags are colored following their labelling status: red for negative (non topic-related), green for positive (topic-related), blue for unknown, and yellow to distinguish the currently observed hashtag. ",
            )
            hashtag_neighbours = (
                st.session_state.keyselect.get_weighted_hashtag_neighbours(
                    observed_hashtag, ht_dict, th_dict
                )
            )

            neighbour_graph = nx.Graph()
            # add all nodes from observed hashtag neighbourhood
            for neighbour in hashtag_neighbours:
                if neighbour not in neighbour_graph:
                    neighbour_graph.add_node(neighbour)
                # get all neighbours of neighbours
                neighbour_neighbourhood = (
                    st.session_state.keyselect.get_weighted_hashtag_neighbours(
                        neighbour,
                        ht_dict,
                        th_dict,
                        restricted_set=set(hashtag_neighbours.keys()),
                    )
                )
                for snd_neighbour in neighbour_neighbourhood:
                    if snd_neighbour not in neighbour_graph:
                        neighbour_graph.add_node(snd_neighbour)

                    if snd_neighbour != neighbour and not neighbour_graph.has_edge(
                        neighbour, snd_neighbour
                    ):
                        neighbour_graph.add_edge(
                            neighbour,
                            snd_neighbour,
                            weight=neighbour_neighbourhood[snd_neighbour],
                        )

            # compute network layout for displayed hashtag
            layout = nx.spring_layout(
                neighbour_graph,
                weight="weight",
            )

            edge_x = []
            edge_y = []
            degrees = []
            for edge in neighbour_graph.edges():
                x0, y0 = layout[edge[0]]
                x1, y1 = layout[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
                degrees.append(
                    neighbour_graph.get_edge_data(edge[0], edge[1])["weight"]
                )

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.1, color="#888"),
                mode="lines",
            )

            node_x = []
            node_y = []
            marker_color = []
            for node in neighbour_graph.nodes():
                x, y = layout[node]
                node_x.append(x)
                node_y.append(y)
                if node in st.session_state.keyselect.baseline_keywords:
                    node_label = "green"
                elif node in st.session_state.keyselect.negative_keywords:
                    node_label = "red"
                elif node == observed_hashtag:
                    node_label = "yellow"
                else:
                    node_label = "blue"
                marker_color.append(node_label)

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    showscale=False,
                    color=marker_color,
                    size=6,
                    line_width=0,
                ),
            )
            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(neighbour_graph.adjacency()):
                # check if node is positive/negative label
                node_name = list(neighbour_graph.nodes())[node]
                node_text.append(
                    f"{node_name} - # of connections: {len(adjacencies[1])}"
                )

            node_trace.text = node_text

            network_fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode="closest",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(b=5, l=5, r=5, t=5),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            )
            col2.plotly_chart(network_fig, use_container_width=True)


if "date_slider" in st.session_state:
    slider_idx = st.session_state.keyselect.ind_days.index(st.session_state.date_slider)
    current_idx = st.session_state.keyselect.iter_days - 1
    if slider_idx != current_idx:
        st.session_state.keyselect.remaining_budget = 0
        st.session_state.keyselect.iter_days = slider_idx


with st.expander("Select New Keywords"):
    if (
        "keyselect" not in st.session_state
        or "seed_keywords" not in st.session_state
        or not topic
    ):
        st.warning("Complete setting up your keyword selection using the sidebar.")
    else:
        md = st.empty()
        # conditions for DONE state: exhausted budget, no more keywords, no more days
        if (
            st.session_state.keyselect.topic_queue is not None
            and st.session_state.keyselect.topic_queue.qsize() == 0
        ) and st.session_state.keyselect.iter_days > len(
            st.session_state.keyselect.ind_days
        ):
            md.markdown(
                f"<h3 style='text-align: center;'>All available tweet-hashtag relations have been processed.</h3>",
                unsafe_allow_html=True,
            )
        else:
            if (
                # need to initialize process
                (
                    st.session_state.keyselect.iter_days == 0
                    and st.session_state.keyselect.remaining_budget
                    == st.session_state.keyselect.budget
                    and st.session_state.keyselect.topic_queue is None
                )
                # exhausted all daily candidate keywords
                or (
                    st.session_state.keyselect.topic_queue is not None
                    and st.session_state.keyselect.topic_queue.qsize() == 0
                )
                # exhausted daily budget
                or st.session_state.keyselect.remaining_budget == 0
            ):
                # init the queue to starting state on first day
                while (
                    st.session_state.keyselect.topic_queue is None
                    or (
                        st.session_state.keyselect.topic_queue.qsize() == 0
                        and st.session_state.keyselect.iter_days
                        <= len(st.session_state.keyselect.ind_days)
                    )
                    or st.session_state.keyselect.remaining_budget == 0
                ):
                    # skip through days until we have some keywords
                    st.session_state.next_keyword = None
                    st.session_state.keyselect.set_budget(labelling_budget)
                    st.session_state.keyselect.get_next_candidates()

            date_slider = md.select_slider(
                "Labelling date",
                options=st.session_state.keyselect.ind_days,
                value=st.session_state.keyselect.ind_days[
                    st.session_state.keyselect.iter_days - 1
                ],
                disabled=False,
                key="date_slider",
            )

            row1a, row2a, row3a, row4a = st.columns([4, 2, 1, 1], gap="medium")
            st.text("")

            col1_, col2_ = st.columns(2, gap="medium")
            prompt = row1a.empty()
            keyword_prompt = row2a.empty()
            do_positive_label = row3a.button("yes 👍")
            if do_positive_label:
                st.session_state["seed_keywords"].append(st.session_state.next_keyword)
                json_keywords_display.json(
                    {
                        "positive": st.session_state.seed_keywords,
                        "negative": list(st.session_state.keyselect.negative_keywords),
                    }
                )
                st.session_state.keyselect.baseline_keywords.add(
                    st.session_state.next_keyword
                )
                st.session_state.keyselect.update_queue_on_keyword(
                    st.session_state.next_keyword
                )
                # decrease counter for current labelling step
                st.session_state.keyselect.remaining_budget -= 1
                st.session_state.labelled_count += 1
                (
                    score,
                    st.session_state.next_keyword,
                ) = st.session_state.keyselect.topic_queue.get_nowait()

            do_negative_label = row4a.button("no 👎")
            if do_negative_label:
                st.session_state.keyselect.negative_keywords.add(
                    st.session_state.next_keyword
                )
                json_keywords_display.json(
                    {
                        "positive": st.session_state.seed_keywords,
                        "negative": list(st.session_state.keyselect.negative_keywords),
                    }
                )

                st.session_state.keyselect.remaining_budget -= 1
                st.session_state.labelled_count += 1
                (
                    score,
                    st.session_state.next_keyword,
                ) = st.session_state.keyselect.topic_queue.get_nowait()

            if st.session_state.next_keyword is None:
                (
                    score,
                    st.session_state.next_keyword,
                ) = st.session_state.keyselect.topic_queue.get_nowait()

            prompt.markdown(
                f"<h4>Is this hashtag relevant to <strong style='font-weight:bold;'>{topic}</strong>?</h4>",
                unsafe_allow_html=True,
            )
            keyword_prompt.markdown(
                f"<h4 style='text-align: center; padding-bottom: 10px; border-style: solid;'><strong style='font-weight:900; color:#EC5953;'>{st.session_state.next_keyword}</strong></h4>",
                unsafe_allow_html=True,
            )

            st.session_state["updated_keyselect_budget"] = col1_.number_input(
                "Remaining budget",
                value=int(st.session_state.keyselect.remaining_budget),
                help="Set the remaining budget for the keyword selection process.",
            )
            col1_.metric(
                "Daily tweet count",
                len(st.session_state.keyselect.tweet_hashtag_dict),
            )

            # add a figure with trend correlations to labelled keywords
            if st.session_state.keyselect.iter_days >= 3:
                (
                    keywords,
                    correlations,
                    labels,
                ) = st.session_state.keyselect.get_trend_correlations(
                    st.session_state.next_keyword
                )

                correlation_dict = {
                    "Labelled keyword": keywords,
                    "Pearson correlation": correlations,
                    "label": labels,
                }
                correlation_df = pd.DataFrame(correlation_dict)
                correlation_fig = px.scatter(
                    correlation_df,
                    x="Labelled keyword",
                    y="Pearson correlation",
                    color="label",
                    title="Pearson correlation of candidate and labelled keyword trends",
                    hover_data=["Labelled keyword"],
                )
                correlation_fig.update_layout(
                    yaxis=dict(showgrid=False, zeroline=True),
                )
                correlation_fig.update_traces(
                    marker=dict(
                        size=14,
                    ),
                    selector=dict(mode="markers"),
                )
                st.plotly_chart(correlation_fig, use_container_width=True)
            else:
                st.info(
                    f"Go through {2-st.session_state.keyselect.iter_days} days of data to show trend correlations."
                )

            # add a graph showing the raw tweet count for the current candidate keyword
            if st.session_state.keyselect.iter_days > 1:
                trend = st.session_state.keyselect.get_trend(
                    st.session_state.next_keyword
                )
                keyword_trend_df = pd.DataFrame.from_dict(trend)
                fig = px.line(
                    keyword_trend_df,
                    x="Date",
                    y="Occurence",
                    title="Hashtag occurence count",
                )
                col1_.plotly_chart(fig, use_container_width=True)
            else:
                col1_.info("Explore more than one day to see the trend.")

            # nodes = []
            # nodes.append(
            #    Node(
            #        id="candidate",
            #        title=f"Tweeted {len(st.session_state.keyselect.hashtag_tweet_dict[st.session_state.next_keyword])} times",
            #        label=st.session_state.next_keyword,
            #        size=25,
            #        shape="dot",
            #    )
            # )
            # added_users = set()
            #
            # edges = []
            # for tid in st.session_state.keyselect.hashtag_tweet_dict[
            #    st.session_state.next_keyword
            # ]:
            #    user_profile_pic = (
            #        st.session_state.keyselect.get_author_user_profile_pic(tid)
            #    )
            #    username = st.session_state.keyselect.get_author_username(tid)
            #    # accept a maximum of 32 users to display
            #    if (
            #        not user_profile_pic
            #        or not username
            #        or username in added_users
            #        or len(nodes) > 8
            #    ):
            #        continue
            #
            #    added_users.add(username)
            #    nodes.append(
            #        Node(
            #            id=username,
            #            title=f"Linked through {make_tweet_url(tid)}",
            #            label=username,
            #            size=25,
            #            shape="circularImage",
            #            image=user_profile_pic,
            #        )
            #    )
            #    edges.append(
            #        Edge(
            #            source=username,
            #            target="candidate",
            #        )
            #    )
            #
            # config = Config(
            #    height=400,
            #    width=400,
            #    directed=False,
            #    collapsible=True,
            #    nodeHighlightBehavior=False,
            #    highlightColor="#EC5953",
            # )
            #
            # with col1_:
            #    st.write("Network of sample users tweeting this hashtag")
            #    return_value = agraph(nodes=nodes, edges=edges, config=config)
            #
            col2_.markdown(
                f"Example tweet for hashtag <strong style='font-weight:bold; color:#EC5953;'>{st.session_state.next_keyword}</strong>",
                unsafe_allow_html=True,
            )
            with col2_:
                exhausted_options = False
                tried_tids = set()
                while not exhausted_options:
                    try:
                        (
                            representative_tid,
                            exhausted_options,
                        ) = st.session_state.keyselect.find_representative_tweet(
                            st.session_state.next_keyword, tried_tids=tried_tids
                        )
                        tried_tids.add(representative_tid)
                        tweet_url = (
                            f"https://twitter.com/any/status/{representative_tid}"
                        )
                        t = Tweet(tweet_url).component()
                        exhausted_options = True
                    except Exception as e:
                        print(e, "tweet was not available", representative_tid)
                        if exhausted_options:
                            col2_.error(
                                "Impossible to find a tweet preview for this hashtag."
                            )

                col2_.markdown(
                    "*Note that a tweet-hashtag relation may come from a retweet of/reply to the displayed tweet, thus the tweet's date may be different to the labelling date shown above.*"
                )
