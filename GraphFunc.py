import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
plt.style.use('seaborn-colorblind')


def drawing_graph(episode_number, average_scores_100, highestscores, gamma, alpha, epsilon_dec):
    # self.pie_graph_number = np.zeros(
    #     (1 if self.scores_pie_graph.size == 0 else self.scores_pie_graph.max().astype(int) + 1))
    # for i in range(0,
    #                (1 if self.scores_pie_graph.size == 0 else self.scores_pie_graph.max().astype(int) + 1)):
    #     for k in range(0, len(self.scores_pie_graph)):
    #         if i == self.scores_pie_graph[k]:
    #             self.pie_graph_number[i] += 1

    # labels = np.arange(0, (
    #     1 if self.scores_pie_graph.size == 0 else self.scores_pie_graph.max().astype(int) + 1))
    # x = self.pie_graph_number[np.where(self.pie_graph_number > 0)]
    # labels = labels[np.where(self.pie_graph_number > 0)].astype(str)
    # explode = ((0.1),) * len(x)
    # normalize = False if self.scores_pie_graph.size == 0 else True
    # plt.pie(x, labels=labels, autopct='%1.1f%%', explode=explode, shadow=False, startangle=90,
    #         normalize=normalize)
    # plt.title("Scores Percentage in Per Episodes" + "\nEpisode Number= " + str(
    #     len(self.scores_pie_graph)) + "," + " Step Number= " + str(ddqn_agent.t), fontsize=10)

    # plt.savefig("/content/drive/MyDrive/deneme/grafik/Pie_Graph" + str(self.graph_counter) + ".pdf")

    textstr = '\n'.join((
        'Gamma' + r'$(\gamma)=%.2f$' % (gamma,),
        'Learning Rate ' + r'$(\alpha)=%f$' % (alpha,),
        'Epsilon Decay ' + r'$(\epsilon)=%f$' % (epsilon_dec,)))

    props = dict(boxstyle='square', facecolor='wheat', alpha=0.75)

    # plt.figure(figsize=(1, 1))
    f, ax = plt.subplots(1)
    ax.set_facecolor("white")
    ax.grid(True)
    # self.plot_counter += 1
    ax.plot(np.arange(1, episode_number + 1), average_scores_100, 'r-', 2)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=1)
    a = list(plt.xticks()[0])
    # a.remove(min(a, key=lambda x: abs(x-np.argmax(average_scores_100))))
    # a.remove(min(a, key=lambda x: abs(x-len(average_scores_100))))
    plt.xticks(a)
    # plt.xticks(a + [np.argmax(average_scores_100)] + [len(average_scores_100)], rotation=90)
    # plt.margins(0.2)
    # plt.subplots_adjust(bottom=0.15)
    b = list(plt.yticks()[0])
    # b.remove(min(b, key=lambda x: abs(x-max(average_scores_100))))
    # b.remove(min(b, key=lambda x: abs(x-average_scores_100[-1])))
    # plt.yticks(b + [max(average_scores_100)] + [average_scores_100[-1]])
    plt.yticks(b)
    plt.ylabel('Average Scores')
    plt.xlabel('Episode Number')

    ax.text(0.75, 0, textstr, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', horizontalalignment='left', bbox=props)

    # plt.savefig("graphs/Average_Scores" + str(self.graph_counter) + ".pdf")
    plt.savefig("graphs/Average_Scores" +
                "_Gamma=" + str(gamma) +
                "_LR=" + str(alpha) +
                "_EpsDec=" + str(epsilon_dec) +
                "_Episode=" + str(episode_number) +
                ".pdf")

    plt.ylabel(' ')
    plt.xlabel(' ')

    # plt.figure(figsize=(1, 1))
    f, ax = plt.subplots(1)
    ax.set_facecolor("white")
    ax.grid(False)
    ax.plot(np.arange(1, episode_number + 1), highestscores, 'r-', 2)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=1)
    a = list(plt.xticks()[0])
    # a.remove(min(a, key=lambda x: abs(x-len(highestscores))))
    plt.xticks(a)
    # plt.xticks(a + [len(highestscores)], rotation=90)
    # plt.margins(0.2)
    # plt.subplots_adjust(bottom=0.15)
    b = list(plt.yticks()[0])
    # b.remove(min(b, key=lambda x: abs(x-highestscores[-1])))
    # plt.yticks(b + [highestscores[-1]])
    plt.yticks(b)
    plt.ylabel('Highest Scores')
    plt.xlabel('Episode Number')

    ax.text(0.75, 0, textstr, transform=ax.transAxes, fontsize=7,
            verticalalignment='bottom', horizontalalignment='left', bbox=props)

    # plt.savefig("graphs/Highest_Scores" + str(self.graph_counter) + ".pdf")
    plt.savefig("graphs/Highest_Scores" +
                "_Gamma=" + str(gamma) +
                "_LR=" + str(alpha) +
                "_EpsDec=" + str(epsilon_dec) +
                "_Episode=" + str(episode_number) +
                ".pdf")

    plt.ylabel(' ')
    plt.xlabel(' ')

    # f, ax = plt.subplots(1)

    # self.graph_counter += 1

