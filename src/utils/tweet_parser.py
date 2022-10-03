import datetime
from typing import Union

TWITTER_API_V1_STR_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
TWITTER_API_ACADEMIC_V2_STR_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
TWITTER_API_V1_STR_FORMAT_OTHER = "%a %b %d %H:%M:%S %z %Y"


class BaseRelationParser:
    def __init__(
        self,
    ):
        self.relation_types = ["mentions", "hashtags", "retweets", "replies", "quotes"]
        self.fn_extract_map = {
            k: v
            for (k, v) in zip(
                self.relation_types,
                [
                    self.get_mention_relations,
                    self.get_hashtag_relations,
                    self.get_retweet_relations,
                    self.get_reply_relations,
                    self.get_quote_relations,
                ],
            )
        }

    def get_creation_time_stamp(self, ts: Union[dict, str]) -> datetime.datetime:
        # $date to parse date object from MongoDB document
        if isinstance(ts, dict) and "$date" in ts:
            return datetime.datetime.strptime(
                ts.get("$date"), TWITTER_API_V1_STR_FORMAT
            )
        elif isinstance(ts, str):
            return datetime.datetime.strptime(ts, TWITTER_API_ACADEMIC_V2_STR_FORMAT)
        else:
            raise ValueError("unrecognized timestamp format")

    def parse_relations_from_tweet(self, tweet: dict):
        output = {}
        for relation_type in self.relation_types:
            tmp_relations = self.fn_extract_map.get(relation_type)(tweet)
            output[relation_type] = tmp_relations

        # retrieve the date
        days = [self.get_creation_time_stamp(tweet["created_at"]).date()]
        if "quoted_status" in tweet:
            days.append(
                self.get_creation_time_stamp(
                    tweet["quoted_status"]["created_at"]
                ).date()
            )
        if "retweeted_status" in tweet:
            days.append(
                self.get_creation_time_stamp(
                    tweet["retweeted_status"]["created_at"]
                ).date()
            )
            if "quoted_status" in tweet["retweeted_status"]:
                days.append(
                    self.get_creation_time_stamp(
                        tweet["retweeted_status"]["quoted_status"]["created_at"]
                    ).date()
                )

        return output, days

    def get_hashtag(self, hashtag: dict) -> str:
        pass

    def get_author_id(self, tweet: dict) -> str:
        pass

    def get_tweet_id(self, tweet: dict) -> str:
        pass

    def get_user_screen_name(self, tweet: dict) -> str:
        return None

    def get_mentioned_user_id(self, mention: dict) -> str:
        pass

    def get_mentioned_user_screen_name(self, mention: dict) -> str:
        pass

    def get_hashtag_relation(
        self, tweet: dict, hashtag: dict, created_at: datetime.datetime
    ) -> dict:
        return {
            "user_id": self.get_author_id(tweet),
            "hashtag": self.get_hashtag(hashtag),
            "tid": self.get_tweet_id(tweet),
            "created_at": created_at,
            "user_screen_name": self.get_user_screen_name(tweet),
        }

    def get_hashtag_relations(self, tweet: dict) -> list:
        hashtag_relations = []
        created_at = self.get_creation_time_stamp(tweet["created_at"])
        if "entities" in tweet and "hashtags" in tweet["entities"]:
            for hashtag in tweet["entities"]["hashtags"]:
                hashtag_relations.append(
                    self.get_hashtag_relation(tweet, hashtag, created_at)
                )

        return hashtag_relations

    def get_mention_relation(
        self, tweet: dict, mention: dict, created_at: datetime.datetime
    ) -> dict:
        return {
            "user_id": self.get_author_id(tweet),
            "mentionned_user_screen_name": self.get_mentioned_user_screen_name(mention),
            "mentionned_user_id": self.get_mentioned_user_id(mention),
            "tid": self.get_tweet_id(tweet),
            "created_at": created_at,
            "user_screen_name": self.get_user_screen_name(tweet),
        }

    def get_mention_relations(self, tweet: dict, key: str = "mentions") -> list:
        mention_relations = []
        created_at = self.get_creation_time_stamp(tweet["created_at"])
        if "entities" in tweet and key in tweet["entities"]:
            for mention in tweet["entities"][key]:
                mention_relations.append(
                    self.get_mention_relation(tweet, mention, created_at)
                )

        return mention_relations

    def get_retweet_author_id(self, tweet: dict):
        pass

    def get_retweet_user_screen_name(self, tweet: dict):
        pass

    def get_retweet_id(self, retweet: dict):
        pass

    def get_retweet_relation(
        self, tweet: dict, retweet: dict, created_at: datetime.datetime
    ) -> dict:
        return {
            "user_id": self.get_author_id(tweet),
            "retweeted_user_id": self.get_retweet_author_id(tweet),
            "retweeted_user_screen_name": self.get_retweet_user_screen_name(tweet),
            "retweet_id": self.get_retweet_id(retweet),
            "tid": self.get_tweet_id(tweet),
            "created_at": created_at,
            "user_screen_name": self.get_user_screen_name(tweet),
        }

    def get_retweet_relations(self, tweet: dict) -> list:
        pass

    def get_reply_relations(self, tweet: dict) -> list:
        pass

    def get_reply_tweet_id(self, reply: dict) -> str:
        pass

    def get_reply_screen_name(self, tweet: dict) -> list:
        pass

    def get_reply_author_id(self, tweet: dict) -> list:
        pass

    def get_quote_screen_name(self, tweet: dict, quote: dict) -> str:
        pass

    def get_quote_tweet_id(self, quote: dict) -> str:
        pass

    def get_quote_author_id(self, quote: dict) -> str:
        # no author id for quoted users with academic api
        return None

    def get_quote_relation(
        self, tweet: dict, quote: dict, created_at: datetime.datetime
    ) -> dict:
        return {
            "user_id": self.get_author_id(tweet),
            "quoted_tweet_id": self.get_quote_tweet_id(quote),
            "tid": self.get_tweet_id(tweet),
            "created_at": created_at,
            "user_screen_name": self.get_user_screen_name(tweet),
            "quoted_user_id": self.get_quote_author_id(quote),
            "quoted_user_screen_name": self.get_quote_screen_name(tweet, quote),
        }

    def get_reply_relation(
        self, tweet: dict, reply: dict, created_at: datetime.datetime
    ) -> dict:
        return {
            "in_reply_to_user_id": self.get_reply_author_id(tweet),
            "in_reply_to_screen_name": self.get_reply_screen_name(tweet),
            "created_at": created_at,
            "user_id": self.get_author_id(tweet),
            "tid": self.get_tweet_id(tweet),
            "user_screen_name": self.get_user_screen_name(tweet),
            "in_reply_to_tweet_id": self.get_reply_tweet_id(reply),
        }

    def get_quote_relations(self, tweet: dict) -> list:
        pass


class AcademicRelationParser(BaseRelationParser):
    def __init__(self):
        super().__init__()

    def get_tweet_id(self, tweet: dict) -> str:
        return tweet.get("id")

    def get_author_id(self, tweet: dict) -> str:
        return tweet.get("author_id")

    def get_hashtag(self, hashtag: dict) -> str:
        return hashtag.get("tag")

    def get_mentioned_user_id(self, mention: dict) -> str:
        return mention.get("id")

    def get_mentioned_user_screen_name(self, mention: dict) -> str:
        return mention.get("username")

    def get_retweet_id(self, retweet: dict) -> str:
        return retweet.get("id")

    def get_reply_tweet_id(self, reply: dict) -> str:
        return reply.get("id")

    def get_reply_screen_name(self, tweet: dict) -> list:
        try:
            return tweet["text"].split("@")[1].split(" ")[0]
        except IndexError:
            # author is replying to themselves so no mentions, and no username
            return None

    def get_mentions_exist(self, tweet: dict) -> bool:
        return "entities" in tweet and "mentions" in tweet["entities"]

    def get_reply_author_id(self, tweet: dict) -> list:
        try:
            in_reply_to_username = tweet["text"].split("@")[1].split(" ")[0]
            if self.get_mentions_exist(tweet):
                for mention in tweet["entities"]["mentions"]:
                    if mention["username"] == in_reply_to_username:
                        return mention["id"]
            else:
                # no mentions, perhaps the author changed their username
                return None
        except IndexError:
            # author is mentioning themselves, is the author id
            return tweet.get("author_id")

    def get_retweet_author_id(self, tweet: dict) -> str:
        retweeted_username = self.get_retweeted_username_from_text(tweet["text"])
        if self.get_mentions_exist(tweet):
            for mention in tweet["entities"]["mentions"]:
                if mention["username"] == retweeted_username:
                    return mention["id"]
        else:
            # no mentions, perhaps the author changed their username
            return None

    def get_retweeted_username_from_text(self, text: str) -> str:
        # extract the first username retweeted from the text
        return text.split("RT @")[1].split(":")[0]

    def get_retweet_user_screen_name(self, tweet: dict):
        retweeted_username = self.get_retweeted_username_from_text(tweet["text"])
        if self.get_mentions_exist(tweet):
            for mention in tweet["entities"]["mentions"]:
                if mention["username"] == retweeted_username:
                    return mention["username"]
        else:
            # no mentions, perhaps the author changed their username
            return retweeted_username

    def get_reply_relations(self, tweet: dict) -> list:
        reply_relations = []
        if "referenced_tweets" in tweet:
            created_at = super().get_creation_time_stamp(tweet["created_at"])
            for referenced_tweet in tweet["referenced_tweets"]:
                if referenced_tweet["type"] == "replied_to":
                    reply_relations.append(
                        self.get_reply_relation(tweet, referenced_tweet, created_at)
                    )

        return reply_relations

    def get_quote_screen_name(self, tweet: dict, quote: dict) -> str:
        for url in tweet["entities"]["urls"]:
            if quote["id"] in url["expanded_url"]:
                quoted_username = (
                    url["expanded_url"].split("https://twitter.com/")[1].split("/")[0]
                )
                return quoted_username
        return None

    def get_quote_tweet_id(self, quote: dict) -> str:
        return quote.get("id")

    def get_quote_relations(self, tweet: dict) -> list:
        quote_relations = []
        if "referenced_tweets" in tweet:
            created_at = super().get_creation_time_stamp(tweet["created_at"])
            for referenced_tweet in tweet["referenced_tweets"]:
                if referenced_tweet["type"] == "quoted":
                    quote_relations.append(
                        self.get_quote_relation(tweet, referenced_tweet, created_at)
                    )

        return quote_relations

    def get_retweet_relations(self, tweet: dict) -> list:
        retweet_relations = []
        created_at = super().get_creation_time_stamp(tweet["created_at"])
        if "referenced_tweets" in tweet:
            for referenced_tweet in tweet["referenced_tweets"]:
                if referenced_tweet.get("type") == "retweeted":
                    retweet_relations.append(
                        self.get_retweet_relation(tweet, referenced_tweet, created_at)
                    )
        return retweet_relations


class StandardRelationParser(BaseRelationParser):
    def __init__(
        self,
    ):
        super().__init__()

    def get_tweet_id(self, tweet: dict) -> str:
        return tweet.get("id_str")

    def get_user_screen_name(self, tweet: dict) -> str:
        return tweet.get("user").get("screen_name")

    def get_author_id(self, tweet: dict) -> str:
        return tweet.get("user").get("id_str")

    def get_hashtag(self, hashtag: dict) -> str:
        return hashtag.get("text")

    def get_mentioned_user_id(self, mention: dict) -> str:
        return mention.get("id_str")

    def get_mentioned_user_screen_name(self, mention: dict) -> str:
        return mention.get("screen_name")

    def get_retweet_author_id(self, tweet: dict) -> str:
        return tweet["retweeted_status"]["user"]["id_str"]

    def get_retweet_user_screen_name(self, tweet: dict) -> str:
        return tweet["retweeted_status"]["user"]["screen_name"]

    def get_retweet_id(self, retweet: dict):
        return retweet["id_str"]

    def get_reply_tweet_id(self, reply: dict) -> str:
        return reply["in_reply_to_status_id_str"]

    def get_reply_screen_name(self, tweet: dict) -> list:
        return (
            None
            if "in_reply_to_screen_name" not in tweet
            else tweet["in_reply_to_screen_name"]
        )

    def get_reply_author_id(self, tweet: dict) -> list:
        return tweet["in_reply_to_user_id_str"]

    def get_quote_screen_name(self, tweet: dict, quote: dict) -> str:
        return quote["user"]["screen_name"]

    def get_quote_tweet_id(self, quote: dict) -> str:
        return quote["id_str"]

    def get_quote_author_id(self, quote: dict) -> str:
        return quote["user"]["id_str"]

    def get_users_from_tweet(self, tweet: dict) -> list:
        users = list()
        users.append(tweet["user"])
        if "quoted_status" in tweet:
            users.append(tweet["quoted_status"]["user"])

        if "retweeted_status" in tweet:
            users.append(tweet["retweeted_status"]["user"])
            if "quoted_status" in tweet["retweeted_status"]:
                users.append(tweet["retweeted_status"]["quoted_status"]["user"])

        return users

    def get_creation_time_stamp(self, ts: Union[dict, str]) -> datetime.datetime:
        if isinstance(ts, str):
            try:
                return datetime.datetime.strptime(ts, TWITTER_API_V1_STR_FORMAT)
            except ValueError:
                return datetime.datetime.strptime(ts, TWITTER_API_V1_STR_FORMAT_OTHER)
        elif isinstance(ts, dict):
            try:
                return datetime.datetime.strptime(
                    ts.get("$date"), TWITTER_API_V1_STR_FORMAT
                )
            except ValueError:
                return datetime.datetime.strptime(
                    ts.get("$date"), TWITTER_API_V1_STR_FORMAT_OTHER
                )
        else:
            raise ValueError("unrecognized timestamp format")

    def get_quotes(self, tweet: dict) -> list:
        quote_relations = []
        if "quoted_status" in tweet:
            created_at = self.get_creation_time_stamp(tweet["created_at"])
            quote_relations.append(
                self.get_quote_relation(tweet, tweet["quoted_status"], created_at)
            )

        return quote_relations

    def get_quote_relations(self, tweet: dict) -> list:
        quote_relations = []
        quote_relations.extend(self.get_quotes(tweet))
        if "retweeted_status" in tweet:
            quote_relations.extend(self.get_quotes(tweet["retweeted_status"]))

        return quote_relations

    def get_replies(self, tweet: dict) -> list:
        reply_relations = []
        if tweet["in_reply_to_status_id"] and tweet["in_reply_to_user_id"]:
            created_at = self.get_creation_time_stamp(tweet["created_at"])
            reply_relations.append(self.get_reply_relation(tweet, tweet, created_at))

        return reply_relations

    def get_reply_relations(self, tweet: dict) -> list:
        reply_relations = []
        reply_relations.extend(self.get_replies(tweet))
        if "quoted_status" in tweet:
            reply_relations.extend(self.get_replies(tweet["quoted_status"]))
        if "retweeted_status" in tweet:
            reply_relations.extend(self.get_replies(tweet["retweeted_status"]))
            if "quoted_status" in tweet["retweeted_status"]:
                reply_relations.extend(
                    self.get_replies(tweet["retweeted_status"]["quoted_status"])
                )

        return reply_relations

    def get_hashtag_relations(self, tweet: dict) -> list:
        hashtag_relations = []
        hashtag_relations.extend(super().get_hashtag_relations(tweet))
        if "quoted_status" in tweet:
            hashtag_relations.extend(
                super().get_hashtag_relations(tweet["quoted_status"])
            )
        if "retweeted_status" in tweet:
            hashtag_relations.extend(
                super().get_hashtag_relations(tweet["retweeted_status"])
            )
            if "quoted_status" in tweet["retweeted_status"]:
                hashtag_relations.extend(
                    super().get_hashtag_relations(
                        tweet["retweeted_status"]["quoted_status"]
                    )
                )

        return hashtag_relations

    def get_mention_relations(self, tweet: dict) -> list:
        """Explore all tweets' structures looking for mentions

        :param tweet: Tweet to process
        :type tweet: dict
        :return: Mention relations found in tweet
        :rtype: list
        """
        mention_relations = []
        mention_relations.extend(
            super().get_mention_relations(tweet, key="user_mentions")
        )
        if "quoted_status" in tweet:
            mention_relations.extend(
                super().get_mention_relations(
                    tweet["quoted_status"], key="user_mentions"
                )
            )
        if "retweeted_status" in tweet:
            mention_relations.extend(
                super().get_mention_relations(
                    tweet["retweeted_status"], key="user_mentions"
                )
            )
            if "quoted_status" in tweet["retweeted_status"]:
                mention_relations.extend(
                    super().get_mention_relations(
                        tweet["retweeted_status"]["quoted_status"], key="user_mentions"
                    )
                )

        return mention_relations

    def parse_metadata(self, tweet: dict) -> dict:
        """Extract tweet metadata for keyselect UI.

        :param tweet: Tweet to be processed
        :type tweet: dict
        :return: Metadata extracted from tweet
        :rtype: dict
        """

        return {
            "retweet_count": tweet.get("retweet_count"),
            "profile_image_url": tweet.get("user", {}).get("profile_image_url"),
            "username": tweet.get("user", {}).get("screen_name"),
        }

    def get_retweet_relations(self, tweet: dict) -> list:
        """Search for retweet relations in tweet

        :param tweet: Tweet to process
        :type tweet: dict
        :return: retweet relations
        :rtype: list
        """
        retweet_relations = []
        created_at = self.get_creation_time_stamp(tweet["created_at"])
        if "retweeted_status" in tweet:
            retweet = tweet["retweeted_status"]
            retweet_relations.append(
                self.get_retweet_relation(tweet, retweet, created_at)
            )

        return retweet_relations
