import networkx as nx
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# Load text files
# words = re.findall(r'\w+', open('pg100.txt').read().lower())
words = re.findall(r'\w+', open('training.eng').read().lower())
words2 = re.findall(r'\w+', open('training.fra').read().lower())


def adjacencymatrix(words):
    ''' 
    This function create a adjacency matrix based on the time series data, e.g, the text
    '''
    wordset=list(set(words))
    i=0
    nodesnum=len(wordset)
    adjacency=np.zeros((nodesnum,nodesnum))
    while i<len(words)-1:
        currentword=words[i]
        forwardword=words[i+1]
        x=wordset.index(currentword)
        y=wordset.index(forwardword)
        adjacency[x][y]=adjacency[x][y]+1
        i=i+1
    return adjacency

def makegraph(words, adjacency,graphtype):
    '''
    This function take both the orginal time serise, and the output of adjacencymatrix() function
    '''
    wordset=list(set(words))
    nodesnum=len(adjacency)
    if graphtype=="normal":
        G=nx.Graph()
        G.add_node("normal")
    elif graphtype=="MultiDi":
        G=nx.MultiDiGraph()
        G.add_node("MultiDi")
    elif graphtype=="Di":
        G=nx.DiGraph()
        G.add_node("Di")
    elif graphtype=="Multi":
        G=nx.MultiGraph()
        G.add_node("Multi")
    else:
        print("no such graph type allowed :(")

    weights=[]
    for x in wordset:
        G.add_node(x)

    for x in range(0,nodesnum):
        for y in range(0, nodesnum):
            if adjacency[x][y] != 0:
                cw=wordset[x]
                fw=wordset[y]
                G.add_weighted_edges_from([(cw,fw,adjacency[x][y])])
                weights.append(adjacency[x][y])
    return G, weights

def plot_basics(data, data_inst, fig, units):
    '''
    This function is the main plotting function. Adapted from Newman's powerlaw package.
    '''
    import pylab
    pylab.rcParams['xtick.major.pad']='8'
    pylab.rcParams['ytick.major.pad']='8'
    pylab.rcParams['font.sans-serif']='Arial'

    from matplotlib import rc
    rc('font', family='sans-serif')
    rc('font', size=10.0)
    rc('text', usetex=False)

    from matplotlib.font_manager import FontProperties

    panel_label_font = FontProperties().copy()
    panel_label_font.set_weight("bold")
    panel_label_font.set_size(12.0)
    panel_label_font.set_family("sans-serif")

    n_data = 1
    n_graphs = 4
    from powerlaw import plot_pdf, Fit, pdf
    ax1 = fig.add_subplot(n_graphs,n_data,data_inst)
    x, y = pdf(data, linear_bins=True)
    ind = y>0
    y = y[ind]
    x = x[:-1]
    x = x[ind]
    ax1.scatter(x, y, color='r', s=.5, label='data')
    plot_pdf(data[data>0], ax=ax1, color='b', linewidth=2, label='PDF')
    from pylab import setp
    setp( ax1.get_xticklabels(), visible=False)
    plt.legend(loc = 'bestloc')

    ax2 = fig.add_subplot(n_graphs,n_data,n_data+data_inst, sharex=ax1)
    plot_pdf(data[data>0], ax=ax2, color='b', linewidth=2, label='PDF')
    fit = Fit(data, discrete=True)
    fit.power_law.plot_pdf(ax=ax2, linestyle=':', color='g',label='w/o xmin')
    p = fit.power_law.pdf()

    ax2.set_xlim(ax1.get_xlim())
    fit = Fit(data, discrete=True,xmin=3)
    fit.power_law.plot_pdf(ax=ax2, linestyle='--', color='g', label='w xmin')
    from pylab import setp
    setp(ax2.get_xticklabels(), visible=False)
    plt.legend(loc = 'bestloc')

    ax3 = fig.add_subplot(n_graphs,n_data,n_data*2+data_inst)#, sharex=ax1)#, sharey=ax2)
    fit.power_law.plot_pdf(ax=ax3, linestyle='--', color='g',label='powerlaw')
    fit.exponential.plot_pdf(ax=ax3, linestyle='--', color='r',label='exp')
    fit.plot_pdf(ax=ax3, color='b', linewidth=2)

    ax3.set_ylim(ax2.get_ylim())
    ax3.set_xlim(ax1.get_xlim())
    plt.legend(loc = 'bestloc')
    ax3.set_xlabel(units)

def fittingplot(G,outputinfo):
    '''
    This is the function we use for creating a degree distribution plot from the complex network 
    '''
    from os import listdir
    blackouts = np.array(list(zip(*dict.items(nx.degree(G)))[1]))
    f = plt.figure(figsize=(8,11))
    data = blackouts
    data_inst = 1
    units = 'Degree distribution of music networks'
    plot_basics(data, data_inst, f, units)
    f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.3, hspace=.2)
    f.savefig('FigWorkflowgraph{0}.png'.format(outputinfo), bbox_inches='tight')

def plotbig(G,outputinfo):
    '''
    This is the function to visualise the network
    '''
    pos=nx.spring_layout(G)   #G is my graph
    # pos=nx.spectral_layout(G)
    plt.figure()
    nx.draw(G,pos,with_labels=True)
    # plt.show()
    # nx.draw(G,pos,node_color='#A0CBE2',edge_color='#BB0000',width=2,edge_cmap=plt.cm.Blues)
    plt.savefig("spectral{0}.png".format(outputinfo), dpi=1000, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1)

# Main function example for degree distribution analysis
# adjacency=adjacencymatrix(words)
# adjacency2=adjacencymatrix(words2)
# G=makegraph(words, adjacency,"normal")[0]
# G2=makegraph(words2, adjacency2,"normal")[0]

# # plotbig(G, 'test')
# fittingplot(G, 'eng')
# fittingplot(G2, 'fr')

def pickhighweight(G0, node):
    '''
    This is the function we use for generating a path of high weight in the network
    '''
    nei = G0.neighbors(node)
    lastweight=0
    candidatenodes=[]
    for it in nei:
        weight=G0[node][it]['weight']
        if weight>=lastweight:
            candidatenodes.append(it)
            lastweight = weight
    pick=random.choice(candidatenodes[-5:])
    return pick

import sys
import random
import numpy as np
def highweightpath(text, outputinfo):
    '''
    This function is to generate new time series based on the high weight path
    '''
    path=[]
    adjacency=adjacencymatrix(text)
    Gtemp=makegraph(text, adjacency,"normal")[0]
    Gcc=sorted(nx.connected_component_subgraphs(Gtemp), key = len, reverse=True)
    G0=Gcc[0]

    tadjacency=np.transpose(adjacency)
    undirectedadja=tadjacency+adjacency
    columnsum=list(undirectedadja.sum(axis=0))
    bigcolumnsum=sorted(list(undirectedadja.sum(axis=0)), reverse=True)[:5]
    print(columnsum)
    rancolumnsum=random.choice(bigcolumnsum)
    print(rancolumnsum)
    firstnote=columnsum.index(rancolumnsum)
    print firstnote
    path.append(firstnote)

    nextnote=random.choice(G0.neighbors(text[firstnote]))
    for i in range (0, 300):
        nextnote=pickhighweight(G0,nextnote)
        path.append(nextnote)

    print(path)
    outFile = open('path/text_{0}.txt'.format(outputinfo), 'w')
    sys.stdout=outFile
    for it in path:
        print(it)
    outFile.close()

# Main function example for creating new time series using the high weight path in the complex network
# highweightpath(words, 'eng')
# highweightpath(words2, 'fr')