from vowpalwabbit import pyvw
import random
from pprint import pprint as P
import numpy as np
from matplotlib import pyplot as plt

seed = random.randint(0, 100000)

m = pyvw.vw(f"--ccb_explore_adf  --cover 5 -q :: --random_seed {seed}")


# We will choose 1 of two users at random for each round with equal probability.
# Each user has 3 features: the user's unique ID (categorical), the time since
# they last logged on (float), and whether or not they are a subscriber (categorical).
# Note that, in VW format, ordinal features (e.g. last_opened:[Float]) need the colon
# separator between the feature name its corresponding value. The colon should not be
# used for categorical features.
user_1_ccb_string = "ccb shared | userid=1 last_opened:0.5 subscriber=true\n"
user_2_ccb_string = "ccb shared | userid=2 last_opened:1.6 subscriber=false\n"

# We next define the reward function based on the selected action (article) for each user.
# For simplicity, we return the reward directly. These rewards could be interpreted
# as the level of engagement of the user, for example the time the user spent reading it.
# For example, rewards of 1.7 and 0.2 could mean that the user spent 1700s and 200s reading
# the article
user_1_reward_dict = {
    "article_1": 0.2,
    "article_2": 0.3,
    "article_3": 0,
    "article_4": 0,
    "article_5": 0,
    "article_6": 0.9,
    "article_7": 1.7,
    "article_8": 0,
    "article_9": 0,
    "article_10": 0,
}
user_2_reward_dict = {
    "article_1": 0.1,
    "article_2": 0.4,
    "article_3": 2.9,
    "article_4": 0,
    "article_5": 0,
    "article_6": 1,
    "article_7": 0.5,
    "article_8": 0,
    "article_9": 0,
    "article_10": 1,
}


def sample_user():
    # choose one of the two users at random
    if random.random() < 0.5:
        return user_1_ccb_string, user_1_reward_dict
    else:
        return user_2_ccb_string, user_2_reward_dict


# simulated user will ignore recommendations after this slot index (reward = 0)
ignore_after_index = 2


def simulate_reward(
    slot_index, chosen_action_features, reward_dict, ignore_after_index
):
    if slot_index > ignore_after_index:
        return 0
    return reward_dict[chosen_action_features]


# define slots in VW CCB format
num_slots = 6
slots_ccb_string = "ccb slot  | \n" * num_slots


action_strings = (
    "article_1",
    "article_2",
    "article_3",
    "article_4",
    "article_5",
    "article_6",
    "article_7",
    "article_8",
    "article_9",
    "article_10",
)  # different actions defined by their features in vw string format


def get_actions_ccb_string(action_strings):
    action_strings = random.sample(action_strings, len(action_strings))
    actions_ccb_string = ""
    for action_string in action_strings:
        actions_ccb_string += f"ccb action | {action_string}\n"
    return actions_ccb_string


# since we know how the users respond to the actions and # of slots,
# we know how best we can do (on average)
best_policy_average_reward = (
    sum(
        sorted(list(user_1_reward_dict.values()), reverse=True)[
            0 : min(ignore_after_index + 1, num_slots)
        ]
    )
    + sum(
        sorted(list(user_2_reward_dict.values()), reverse=True)[
            0 : min(ignore_after_index + 1, num_slots)
        ]
    )
) / 2


def to_ccb_slot_format_result(result):
    """converts the chosen action, cost, and probability tuples for each slot into
    a string for creating the VW CCB update string.
    """
    result = (
        str(
            [
                str(x)
                .replace(" ", "")
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace(",", ":")
                for x in result
            ]
        )
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .replace("'", "")
    )
    return result


def replace_nth_string_occurance(s, source, target, n):
    """Adds the action, cost, and probability string at the right slot index of the predict
    string to create the update string in CCB Input Format.

    The final output for each slot will be of the form
    "ccb slot [<chosen_action>:<cost>:<probability>,<action>:<probability>,...] | ..."
    as is specified in the CCB Input Format (here we assume that there are no slot
    specific action restrictions besides the available actions, so it won't include the
    [action_ids_to_include,...] part shown in the documentation).

    Args:
        s (Str): source string
        source (Str): string occurence to be replaced
        target (Str): string occurrence to be replaced with
        n (Int): index of string occurence to be replaced

    Returns: s with nth string (source) occurence replaced with target

    """
    inds = [
        i for i in range(len(s) - len(source) + 1) if s[i : i + len(source)] == source
    ]
    if len(inds) < n:
        return  # or maybe raise an error
    s = list(s)  # can't assign to string slices. So, let's listify
    s[
        inds[n - 1] : inds[n - 1] + len(source)
    ] = target  # do n-1 because we start from the first occurrence of the string, not the 0-th
    return "".join(s)


# simulation
num_rounds = 500  # we'll simulate 500 rounds

# random policy simulation
round_results_random_policy = []
round_rewards_random_policy = []
for _ in range(num_rounds):
    # sample user
    _, reward_dict = sample_user()
    # choose num_slots actions (without replacement)
    round_result_random_policy = random.sample(action_strings, num_slots)
    # get simulated reward based on chosen actions of random policy
    round_reward_random_policy = sum(
        [
            simulate_reward(
                i, round_result_random_policy[i], reward_dict, ignore_after_index
            )
            for i in range(num_slots)
        ]
    )
    # store result and reward
    round_rewards_random_policy.append(round_reward_random_policy)
    round_results_random_policy.append(round_result_random_policy)

# CCB simulation
round_results_ccb = []
round_rewards_ccb = []
for _ in range(num_rounds):
    # shuffle actions list in VW CCB format to mimic realistic setting of changing actions set
    actions_ccb_string = get_actions_ccb_string(action_strings)
    user_ccb_string, reward_dict = sample_user()
    input_ccb_string = user_ccb_string + actions_ccb_string + slots_ccb_string
    # call model for choosing action for each slot
    round_result_ccb = m.predict(input_ccb_string)
    ccb_round_reward = 0
    # initialize update string
    update_ccb_string = input_ccb_string
    for slot_index in range(len(round_result_ccb)):
        # get simulated reward based on chosen action of bandit model
        chosen_action_index = round_result_ccb[slot_index][0][0]
        chosen_action_features = actions_ccb_string.split("\n")[
            chosen_action_index
        ].split("| ", 1)[1]
        reward = simulate_reward(
            slot_index, chosen_action_features, reward_dict, ignore_after_index
        )
        ccb_round_reward += reward
        # incorporate reward and actions selected into update string
        round_result_ccb[slot_index][0] = (
            round_result_ccb[slot_index][0][0],
            -reward,  # cost = -reward
            round_result_ccb[slot_index][0][1],
        )
        update_ccb_string = replace_nth_string_occurance(
            update_ccb_string,
            "ccb slot",
            "ccb slot " + to_ccb_slot_format_result(round_result_ccb[slot_index]),
            slot_index + 1,
        )
    # update model using fully formulated update string for the round
    m.learn(update_ccb_string)
    # store result and reward
    round_results_ccb.append(round_result_ccb)
    round_rewards_ccb.append(ccb_round_reward)


# compute regret
round_regrets_ccb = best_policy_average_reward - np.array(round_rewards_ccb)
round_regrets_random_policy = best_policy_average_reward - np.array(
    round_rewards_random_policy
)
# plot rolling average regret
roll = 10
rolling_round_regrets_ccb = np.convolve(
    round_regrets_ccb, np.ones(roll) / roll, mode="valid"
)
rolling_round_regrets_random_policy = np.convolve(
    round_regrets_random_policy, np.ones(roll) / roll, mode="valid"
)
print(np.mean(round_regrets_ccb))
plt.plot(
    list(range(len(rolling_round_regrets_ccb))), rolling_round_regrets_ccb, label="ccb"
)
plt.plot(
    list(range(len(rolling_round_regrets_random_policy))),
    rolling_round_regrets_random_policy,
    label="random policy",
)
plt.plot(
    list(range(len(rolling_round_regrets_ccb))),
    [0] * len(rolling_round_regrets_ccb),
)
plt.xlabel("round")
plt.ylabel(f"rolling average regret (N={roll})")
plt.ylim(-1, best_policy_average_reward)
plt.title(
    f"Rolling regret for 10 candidates, 6 slots, user ignores all slots after #{ignore_after_index + 1}"
)
plt.legend(loc="upper left")
if ignore_after_index == 2:
    plt.savefig("sim_0")
elif ignore_after_index == 5:
    plt.savefig("sim_1")
plt.show()
