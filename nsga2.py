import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

# a exemple of multi-obj fouction (SCH)
def get_objectives_SCH(x):
    return np.array([x**2, (x-2)**2]).T


def get_objectives(x):
    return np.array([(x-400)**2, (x-600)**2]).T

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(objectives):
    popSize, objFucNum = objectives.shape
    ps = np.repeat(objectives, popSize, axis=0)
    qs = np.tile(objectives, (popSize, 1))
    # compute n_p
    dominated_flags = np.sum(
        qs-ps < 0, axis=1).reshape(popSize, popSize) == objFucNum
    N = np.sum(dominated_flags, axis=1)
    # compute S_p
    S = []
    dominate_flags = np.sum(
        ps-qs < 0, axis=1).reshape(popSize, popSize) == objFucNum
    for dominate_flag in dominate_flags:
        S_p = []
        for q, flag in enumerate(dominate_flag):
            if flag:
                S_p.append(q)
        S.append(S_p)

    # compute front
    front = []
    while True:
        Fi = np.argwhere(N == 0).reshape(-1)
        if Fi.size == 0:
            break
        front.append(Fi)
        N[Fi] = -1
        for p in Fi:
            for q in S[p]:
                N[q] -= 1
    return front

#Function to calculate crowding distance
def crowding_distance(objectives, front):
    popSize, objFucNum = objectives.shape
    distance = np.zeros(popSize)

    for front_objs_idx in front:
        if len(front_objs_idx) <= 2:
            distance[front_objs_idx] = np.inf

        tmp_dis = np.zeros(len(front_objs_idx))
        front_objs = objectives[front_objs_idx, :]
        ind = np.argsort(front_objs, axis=0).T
        for i in range(objFucNum):
            front_objs_i = front_objs[:, i]
            tmp_dis[ind[i, 0]] = np.inf
            tmp_dis[ind[i, -1]] = np.inf
            extent = front_objs_i[ind[i][-1]] - front_objs_i[ind[i][0]]
            for j in range(1, len(front_objs_idx)-1):
                tmp_dis[ind[i][j]] += (front_objs_i[ind[i][j+1]] -
                                       front_objs_i[ind[i][j-1]]) / extent
        distance[front_objs_idx] = tmp_dis
    return distance

# SBX cross
def SBX_cross(population, cross_prob=0.5, expo=1):
    for _ in range(len(population)):
        if np.random.random() < cross_prob:
            # compute para
            mu = np.random.random()
            if mu < 0.5:
                para = np.power(2*mu, 1/(expo+1))
            else:
                para = np.power(1/(2*(1-mu)), 1/(expo+1))
            # cross
            ind1, ind2 = random.sample(range(len(population)), 2)
            p1, p2 = population[ind1], population[ind2]
            temp1 = p1
            p1 = 0.5*((1+para)*p1 + (1-para)*p2)
            p2 = 0.5*((1-para)*temp1 + (1+para)*p2)
            population[ind1] = p1
            population[ind2] = p2
    return population

def random_cross(population, cross_prob=0.5, expo=1):
    for _ in range(len(population)):
        if np.random.random() < cross_prob:
            # compute para
            mu = np.random.random()
            if mu <0.5:
                para = 1 - mu
            else:
                para = mu
            # cross
            ind1, ind2 = random.sample(range(len(population)), 2)
            p1, p2 = population[ind1], population[ind2]
            temp1 = p1
            p1 = para*p1 + (1-para)*p2
            p2 = (1-para)*temp1 + para*p2
            population[ind1] = p1
            population[ind2] = p2
    return population


# polynomial mutation, 
def poly_mutate(population, mute_prob=0.1, expo=1):
    for _ in range(len(population)):
        if np.random.random() < mute_prob:
            # compute para
            mu = np.random.random()
            if mu < 0.5:
                addition = np.power(2*mu, 1/(expo+1)) - 1
            else:
                addition = np.power(1-(2*(1-mu)), 1/(expo+1))
            # cross
            ind = random.sample(range(len(population)), 1)
            population[ind] += addition
    return population


def random_mutate(population, mute_prob=0.2, _min=-1000, _max=1000):
    for _ in range(len(population)):
        if np.random.random() < mute_prob:
            ind = random.sample(range(len(population)), 1)
            population[ind] = _min + np.random.random() * (_max-_min)
    return population


def plot_front(objectives, front, iterNum, first=None, save=True):
    cmaps = cm.get_cmap('viridis', len(front))
    if first:
        if first < len(front):
            cmaps = cm.get_cmap('viridis', first)
    
    plt.cla()
    for rank, fr in enumerate(front):
        if first:
            if rank > first-1:
                break
        plt.plot(objectives[fr, 0], objectives[fr, 1], 'o--',
                 c=cmaps(rank), label='rank-{}'.format(rank))
    plt.xlabel("obj1")
    plt.ylabel("obj2")
    plt.legend(loc="best", prop={'size': 6})
    plt.title('Pareto set at iter {}'.format(iterNum))
    plt.savefig('./result/exp/pareto_{}.png'.format(iterNum))
    plt.show()

def plot_obj(population, objectives, front, iterNum, first=None, save=True):
    temp_cmaps = cm.get_cmap('viridis', len(front))
    cmaps = [np.expand_dims(np.array(temp_cmaps(i)), axis=0)
             for i in range(len(front))]
    if first:
        if first < len(front):
            temp_cmaps = cm.get_cmap('viridis', first)
            cmaps = [np.expand_dims(np.array(temp_cmaps(i)), axis=0)
                     for i in range(first)]
    
    x = np.arange(0, 1000)
    model_obj = get_objectives(x)
    plt.cla()
    plt.subplot(2, 1, 1)
    plt.title('object 1 value at iter {}'.format(iterNum))
    plt.xlabel('x')
    plt.ylabel('obj1')
    plt.plot(x, model_obj[:, 0])
    for rank, fr in enumerate(front):
        if first:
            if rank > first-1:
                break
        plt.scatter(population[fr], objectives[fr, 0], marker='o',
                    c=cmaps[rank], label='rank-{}'.format(rank))
    plt.legend(loc="best", prop={'size': 6})

    plt.subplot(2, 1, 2)
    plt.title('object 2 value at iter {}'.format(iterNum))
    plt.xlabel('x')
    plt.ylabel('obj2')
    plt.plot(x, model_obj[:, 1])

    for rank, fr in enumerate(front):
        if first:
            if rank > first-1:
                break
        plt.scatter(population[fr], objectives[fr, 1], marker='o',
                    c=cmaps[rank], label='rank-{}'.format(rank))
    plt.legend(loc="best", prop={'size': 6})

    plt.tight_layout()
    plt.savefig('./result/exp/obj_{}.png'.format(iterNum))
    plt.show()


def plot_iter1_obj(population, objectives):
    x = np.arange(0, 1000)
    model_obj = get_objectives(x)
    plt.cla()
    plt.subplot(2, 1, 1)
    #plt.title('init popluatoin')
    plt.xlabel('x')
    plt.ylabel('obj1')
    plt.plot(x, model_obj[:, 0])
    plt.scatter(population, objectives[:, 0], marker='o')

    plt.subplot(2, 1, 2)
    # plt.title('object 2 value at iter {}'.format(iterNum))
    plt.xlabel('x')
    plt.ylabel('obj2')
    plt.plot(x, model_obj[:, 1])
    plt.scatter(population, objectives[:, 1], marker='o')

    plt.tight_layout()
    # plt.savefig('./result/exp/init_pop.png'.format(iterNum))
    plt.show()

if __name__ == "__main__":
    popSize = 10
    iterNum = 10
    iterNumBak = iterNum
    SAVE = True
    SAVE_EVERY = 1
    # initiate
    population = np.random.random(popSize) * 1000
    plot_iter1_obj(population, get_objectives(population))
    print('--------------init pop--------------')
    print(population)
    while iterNum > 0:
        children = population.copy()
        # cross
        children = SBX_cross(children)
        if iterNum == iterNumBak:
            plot_iter1_obj(children, get_objectives(children))
            print('--------------iter1 SBX_cross--------------')
            print(children)
        # mute
        children = poly_mutate(children)
        if iterNum == iterNumBak:
            plot_iter1_obj(children, get_objectives(children))
            print('--------------iter1 poly_mutate--------------')
            print(children)
        # elite select
        population = np.unique(np.concatenate([population, children]))
        if iterNum == iterNumBak:
            plot_iter1_obj(population, get_objectives(population))
            print('--------------iter1 elite save--------------')
            print(population)
        # select
        objectives = get_objectives(population)
        front = fast_non_dominated_sort(objectives)
        if (iterNumBak-iterNum) % SAVE_EVERY == 0:
            plot_obj(population, objectives, front,
                    iterNumBak-iterNum, first=5, save=SAVE)
            plot_front(objectives, front, iterNumBak-iterNum, first=5, save=SAVE)
        if iterNum == iterNumBak:
            print('--------------iter1 front--------------')
            print(front)

        distance = crowding_distance(objectives, front)
        if iterNum == iterNumBak:
            print('--------------iter1 distance--------------')
            print(distance)
        next_idx = np.zeros(popSize, dtype=np.int)
        ptr = 0
        for fidx in front:
            if ptr+len(fidx) > popSize:
                if ptr == popSize:
                    break
                # distance
                rank_dis = distance[fidx]
                next_idx[ptr:] = fidx[np.argsort(rank_dis)[ptr-popSize:]]
                break
            else:
                next_ptr = ptr+len(fidx)
                next_idx[ptr:next_ptr] = fidx
                ptr = next_ptr
        population = population[next_idx]
        iterNum -= 1
        if iterNum == iterNumBak:
            print('--------------iter1 select--------------')
            print(population)
    print(population)
