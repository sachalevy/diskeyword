import queue

import scipy
import numpy as np


class KeySelect:
    def __init__(
        self,
        tweet_hashtag_relations: list,
        ind_days: list,
        tweet_metadata: dict = None,
        baseline_keywords: set = None,
    ):
        """Initialize KeySelect object, given a list of tweet-hashtag relations, and initial keywords.

        :param tweet_hashtag_relations: Dict-like objects representing a link between a tweet and a quoted hashtag.
        :type tweet_hashtag_relations: list
        :param ind_days: Set of days to consider when processing data.
        :type ind_days: list
        :param baseline_keywords: Set of initial keywords to sample graph, defaults to None
        :type baseline_keywords: set, optional
        """
        self.tweet_metadata = tweet_metadata
        self.baseline_keywords = baseline_keywords if baseline_keywords else set()
        self.negative_keywords = set()
        self.candidate_keywords = set()
        self.iter_days = 0
        self.budget = 30
        self.topic_queue = None
        self.remaining_budget = self.budget
        self.relation_count = 0
        self.hashtag_count, self.tweet_count, self.user_count = 0, 0, 0
        self.hashtag_idx, self.idx_hashtag = {}, {}
        self.tweet_idx, self.idx_tweet = {}, {}
        self.total_hashag_occurences = {}
        self.tweet_hashtag_dict, self.hashtag_tweet_dict = {}, {}
        self.queued_keywords = set()
        self.load_data(tweet_hashtag_relations, ind_days)

    def load_data(self, tweet_hashtag_relations: list, ind_days: list):
        """Extract network from provided tweet-hashtag relations over the given days.

        :param tweet_hashtag_relations: List of dict-like objects connecting hashtags to their tweets.
        :type tweet_hashtag_relations: list
        :param ind_days: Set of days to include in processing.
        :type ind_days: list
        """
        users = set()
        day_to_relation_count = {}
        for hashtag_relation in tweet_hashtag_relations:
            day = hashtag_relation.get("created_at").date().strftime("%Y-%m-%d")
            if day not in day_to_relation_count:
                day_to_relation_count[day] = {"th_count": 0, "t_count": set()}
            day_to_relation_count[day]["th_count"] += 1
            day_to_relation_count[day]["t_count"].add(hashtag_relation.get("tid"))

        self.daily_tweet_count = {}
        for day in ind_days:
            if day not in day_to_relation_count:
                self.daily_tweet_count[day] = 0
            else:
                self.daily_tweet_count[day] = len(day_to_relation_count[day]["t_count"])

        significant_days = []
        # filter out days with less than 100 tweets, while preserving order
        for day in [
            d
            for d in ind_days
            if d in set(ind_days).intersection(set(day_to_relation_count.keys()))
        ]:
            if (
                day_to_relation_count.get(day)["th_count"] > 100
                and len(day_to_relation_count.get(day)["t_count"]) > 100
            ):
                significant_days.append(day)

        # initialize tweet and hashtag index
        self.ind_days = significant_days
        self.day_idx_dict, self.idx_day_dict = {
            day: i for i, day in enumerate(self.ind_days)
        }, {i: day for i, day in enumerate(self.ind_days)}
        self.day_batches = {i: set() for i in range(len(self.ind_days))}
        self.daily_hashtag_occurences = {i: {} for i in range(len(self.ind_days))}

        # build aggregated hashtag-tweet graph
        for hashtag_relation in tweet_hashtag_relations:
            # trimming down hashtag
            hashtag = hashtag_relation["hashtag"].lower().replace("ー", "-")
            # retrieve day corresponding to tweet creation date
            day = self.day_idx_dict.get(
                hashtag_relation.get("created_at").date().strftime("%Y-%m-%d")
            )
            # skip irrelevant relations
            if day is None or self.idx_day_dict[day] not in self.ind_days:
                continue

            if hashtag not in self.hashtag_idx:
                self.hashtag_idx[hashtag] = self.hashtag_count
                self.idx_hashtag[self.hashtag_count] = hashtag
                self.hashtag_count += 1

            # keep track of all loaded users
            users.add(hashtag_relation["user_id"])

            if hashtag_relation.get("tid") not in self.tweet_idx:
                self.tweet_idx[hashtag_relation.get("tid")] = self.tweet_count
                self.idx_tweet[self.tweet_count] = hashtag_relation.get("tid")
                self.tweet_count += 1

            if hashtag not in self.daily_hashtag_occurences[day]:
                self.daily_hashtag_occurences[day][hashtag] = 0
            self.daily_hashtag_occurences[day][hashtag] += 1

            # expand edge-list
            self.day_batches[day].add(
                (
                    self.hashtag_idx.get(hashtag),
                    self.tweet_idx.get(hashtag_relation.get("tid")),
                )
            )
            self.relation_count += 1

        self.user_count = len(users)

    def __iter__(self):
        """Reset counter upon begining a new iteration."""
        self.iter_days = 0
        self.remaining_budget = self.budget

        return self

    def get_trend(self, keyword):
        counts, dates = [], []
        if keyword not in self.hashtag_idx:
            return {"Date": dates, "Occurence": counts}

        for idx, day in enumerate(self.ind_days[: self.iter_days]):
            dates.append(day)
            if keyword not in self.daily_hashtag_occurences[idx]:
                counts.append(0)
            else:
                counts.append(self.daily_hashtag_occurences[idx][keyword])

        return {"Date": dates, "Occurence": counts}

    def find_representative_tweet(self, keyword: str, tried_tids: set):
        tids = self.hashtag_tweet_dict[keyword]
        # no metadata available, return first tweet as an example
        if not self.tweet_metadata:
            return list(tids)[0]

        # otherwise find the one with highest retweet count
        sorted_tids = sorted(
            {
                tid: self.tweet_metadata[tid]["retweet_count"]
                for tid in tids
                if tid in self.tweet_metadata and tid not in tried_tids
            },
            key=lambda x: x[1],
            reverse=True,
        )
        if len(sorted_tids) == 0:
            return list(tids)[0]

        return sorted_tids[0], len(sorted_tids) <= 1

    def get_author_user_profile_pic(self, tid: str):
        if not self.tweet_metadata or tid not in self.tweet_metadata:
            return None
        return self.tweet_metadata[tid]["profile_image_url"]

    def get_author_username(self, tid: str):
        if not self.tweet_metadata or tid not in self.tweet_metadata:
            return None
        return self.tweet_metadata[tid]["username"]

    def get_evicted_keywords(self) -> dict:
        """Remove hashtags more popular than most baseline keyword with highest degree."""
        evicted_keywords = {}
        # make sure get the most popular ones on this day
        sorted_occurences = sorted(
            self.daily_hashtag_occurences.get(self.iter_days).items(),
            key=lambda x: x[1],
            reverse=True,
        )

        evicted_keywords = set()
        for hashtag, _ in sorted_occurences:
            if hashtag in self.baseline_keywords:
                break
            else:
                evicted_keywords.add(hashtag)

        return evicted_keywords

    def get_hashtag_scores(
        self, hashtag_tweet_dict: dict, positive_labels: set, negative_labels: set
    ):
        """
        Compute the scores for each of these hashtags.
        """

        all_to_labelled_scores = {}
        for hashtag in hashtag_tweet_dict:
            if hashtag in positive_labels or hashtag in negative_labels:
                continue

            pos, pos_coef = 0, 0
            for positive_keyword in positive_labels:
                pos_inter_tids = hashtag_tweet_dict[hashtag].intersection(
                    hashtag_tweet_dict[positive_keyword]
                )
                pos += len(pos_inter_tids)
                pos_coef += 1

            neg, neg_coef = 0, 0
            for negative_keyword in negative_labels:
                neg_inter_tids = hashtag_tweet_dict[hashtag].intersection(
                    hashtag_tweet_dict[negative_keyword]
                )
                neg += len(neg_inter_tids)
                neg_coef += 1

            all_to_labelled_scores[hashtag] = pos / max(1, pos_coef) - neg / max(
                1, neg_coef
            )

        return all_to_labelled_scores

    def __next__(self):
        """Get next batch of tweets, explicitely unrolling 1 loop."""
        # stop when exhausted all days to consider
        if self.iter_days > len(self.ind_days):
            raise StopIteration()
        tweet_hashtag_dict, hashtag_tweet_dict = self.get_next_graph_batch()
        return tweet_hashtag_dict, hashtag_tweet_dict

    def get_next_candidates(self):
        """Initialize candidate keywords."""
        # re-initialize candidate keyword labelling budget
        self.remaining_budget = self.budget
        # generate graph for given day
        self.tweet_hashtag_dict, self.hashtag_tweet_dict = self.get_next_graph_batch()
        available_baseline_keywords = self.baseline_keywords.intersection(
            set(self.hashtag_tweet_dict)
        )
        available_negative_keywords = self.negative_keywords.intersection(
            set(self.hashtag_tweet_dict)
        )
        self.centralities = self.get_hashtag_scores(
            self.hashtag_tweet_dict,
            available_baseline_keywords,
            available_negative_keywords,
        )
        self.topic_queue = queue.PriorityQueue()
        for keyword in available_baseline_keywords:
            self.update_queue_on_keyword(keyword)

    def update_queue_on_keyword(self, keyword: str):
        """Once a keyword has been positively labelled, extend graph to include its neighbours.

        :param keyword: Newly positively labelled keyword
        :type keyword: str
        """
        neighbours = (
            self.get_hashtag_neighbours(
                keyword,
                self.hashtag_tweet_dict,
                self.tweet_hashtag_dict,
            )
            - self.baseline_keywords
            - self.negative_keywords
            - self.queued_keywords
        )
        sorted_neighbours_by_centralities = {
            k: v
            for k, v in sorted(
                self.centralities.items(), key=lambda item: item[1], reverse=True
            )
            if k in neighbours
        }
        for neighbour in list(sorted_neighbours_by_centralities):
            self.queued_keywords.add(neighbour)
            self.topic_queue.put((-self.centralities[neighbour], neighbour))

    def get_hashtag_neighbours(
        self,
        keyword: str,
        hashtag_tweet_dict: dict,
        tweet_hashtag_dict: dict,
    ) -> set:
        """Extract neighbourhood of a given keyword.

        :param keyword: Keyword to process
        :type keyword: str
        :param hashtag_tweet_dict: Hashtag to tweet dictionary to use as graph
        :type hashtag_tweet_dict: dict
        :param tweet_hashtag_dict: Tweet to hashtag dictionary to use as graph
        :type tweet_hashtag_dict: dict
        :return: Set of retrieved neighbours
        :rtype: set
        """
        neighbours = set()
        if keyword not in hashtag_tweet_dict:
            return neighbours

        for tid in hashtag_tweet_dict[keyword]:
            neighbours.update(tweet_hashtag_dict[tid])

        return neighbours

    def get_weighted_hashtag_neighbours(
        self,
        keyword: str,
        hashtag_tweet_dict: dict,
        tweet_hashtag_dict: dict,
        restricted_set: set = None,
    ):
        neighbours = {}
        if keyword not in hashtag_tweet_dict:
            return neighbours

        for tid in hashtag_tweet_dict[keyword]:
            for hashtag in tweet_hashtag_dict[tid]:
                # restrict exploration to a limited set of hashtags
                if restricted_set and hashtag not in restricted_set:
                    continue

                if hashtag not in neighbours:
                    neighbours[hashtag] = 0
                neighbours[hashtag] += 1

        return neighbours

    def get_trend_correlations(self, keyword: str):
        """Explore correlations over days between labelled keywords and a given keyword."""

        make_series = lambda k: np.array(
            [
                self.daily_hashtag_occurences[idx].get(k, 0)
                for idx, _ in enumerate(self.ind_days[: self.iter_days])
            ]
        )
        candidate_series = make_series(keyword)
        keywords, correlations, labels = [], [], []
        for labelled_keywords, label in zip(
            [self.baseline_keywords, self.negative_keywords], ["positive", "negative"]
        ):
            for labelled_keyword in labelled_keywords:
                labelled_series = make_series(labelled_keyword)
                correlation_val = scipy.stats.pearsonr(
                    candidate_series, labelled_series
                ).statistic
                if np.isnan(correlation_val):
                    continue

                keywords.append(labelled_keyword)
                labels.append(label)
                correlations.append(correlation_val)

        return keywords, correlations, labels

    def get_next_graph_batch(self):
        evicted_keywords = self.get_evicted_keywords()
        tweet_hashtag_dict, hashtag_tweet_dict = {}, {}
        self.queued_keywords = set()

        direct_edges = [
            edge
            for edge in self.day_batches[self.iter_days]
            if (
                (self.idx_hashtag[edge[0]] in self.baseline_keywords)
                and (self.idx_hashtag[edge[0]] not in evicted_keywords)
            )
        ]

        for edge in direct_edges:
            if self.idx_hashtag[edge[0]] not in hashtag_tweet_dict:
                hashtag_tweet_dict[self.idx_hashtag[edge[0]]] = set()
            hashtag_tweet_dict[self.idx_hashtag[edge[0]]].add(self.idx_tweet[edge[1]])

            if self.idx_tweet[edge[1]] not in tweet_hashtag_dict:
                tweet_hashtag_dict[self.idx_tweet[edge[1]]] = list()
            tweet_hashtag_dict[self.idx_tweet[edge[1]]].append(
                self.idx_hashtag[edge[0]]
            )

        # expand edges along 1 hop neighbours on hashtag-hashtag graph
        extra_edges = [
            edge
            for edge in self.day_batches[self.iter_days]
            if self.idx_tweet[edge[1]] in tweet_hashtag_dict
            and self.idx_hashtag[edge[0]] not in self.baseline_keywords
            and self.idx_hashtag[edge[0]] not in evicted_keywords
        ]
        for edge in extra_edges:
            if self.idx_hashtag[edge[0]] not in hashtag_tweet_dict:
                # hashtag can be only quoted once in a tweet
                hashtag_tweet_dict[self.idx_hashtag[edge[0]]] = set()
            hashtag_tweet_dict[self.idx_hashtag[edge[0]]].add(self.idx_tweet[edge[1]])

            if self.idx_tweet[edge[1]] not in tweet_hashtag_dict:
                # tweet can quote many hashtags
                tweet_hashtag_dict[self.idx_tweet[edge[1]]] = list()
            tweet_hashtag_dict[self.idx_tweet[edge[1]]].append(
                self.idx_hashtag[edge[0]]
            )

        self.iter_days += 1
        return tweet_hashtag_dict, hashtag_tweet_dict

    def set_baseline_keywords(self, baseline_keywords: set):
        self.baseline_keywords = set(baseline_keywords)

    def set_budget(self, budget: int):
        self.budget = budget
        self.remaining_budget = budget
