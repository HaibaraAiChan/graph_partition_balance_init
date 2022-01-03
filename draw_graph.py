import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def merge_(list1, list2):
	merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
	return merged_list

def draw_graph(G):
    
	fig = plt.figure()
	black_edges = G.edges()
	black_edges = list(black_edges)
	print(black_edges[0])
	print(black_edges[1])
	# print('total eid number   '	)
	# print(len(black_edges[0]))
	# dd = int(len(black_edges[0]) / 2)
	dd = int(len(black_edges[0]))
	black_edges[0] = black_edges[0].tolist()
	black_edges[1] = black_edges[1].tolist()
	black_edges = merge_(black_edges[0][:dd], black_edges[1][:dd])
	# print('black_edges')
	# print(black_edges)
	nx_G = G.to_networkx()
	# nx_G = G.to_networkx().to_undirected()

	# pos = nx.kamada_kawai_layout(nx_G)
	pos = nx.spring_layout(nx_G)

	nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
	# nx.draw_networkx_edge_labels(nx_G, pos, font_color='r', label_pos=0.7)
	# nx.draw_networkx_edges(nx_G, pos,  arrows=False)
	nx.draw_networkx_edges(nx_G, pos, edgelist=black_edges, arrows=True)
	ax = plt.gca()
	ax.margins(0.20)

	plt.axis("off")
	# plt.show()
	plt.savefig('TTTTTTTTT karate full batch sub-graph.eps',format='eps')
	return