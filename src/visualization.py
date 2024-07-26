import matplotlib.pyplot as plt

def visualize_votes(no_user_voted, no_movies_voted):
    f, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.scatter(no_user_voted.index, no_user_voted, color='mediumseagreen')
    plt.axhline(y=10, color='r')
    plt.xlabel('MovieId')
    plt.ylabel('No. of users voted')
    plt.show()

    f, ax = plt.subplots(1, 1, figsize=(16, 4))
    plt.scatter(no_movies_voted.index, no_movies_voted, color='mediumseagreen')
    plt.axhline(y=50, color='r')
    plt.xlabel('UserId')
    plt.ylabel('No. of votes by user')
    plt.show()
