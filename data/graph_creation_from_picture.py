import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as mpimg
from matplotlib.backend_bases import MouseButton

class GraphBuilder:
    def __init__(self, image_path):
        self.image = mpimg.imread(image_path)
        self.graph = nx.Graph()
        self.current_origin = None
        self.node_count = 0

        # Setup plot
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            # Add a new node
            node_name = f"Node_{self.node_count}"
            self.graph.add_node(node_name, pos=(event.xdata, event.ydata))
            if self.current_origin:
                # Connect to the current origin
                self.graph.add_edge(self.current_origin, node_name)
            else:
                # Set as the current origin if it's the first node
                self.current_origin = node_name
            self.node_count += 1

            # Redraw the plot with the new nodes and edges
            self.redraw_plot()

    def on_key(self, event):
        if event.key == ' ':
            # Change the origin node
            print("Please click on a new origin node")
            # The actual implementation for changing the origin node will depend on the user interaction

    def redraw_plot(self):
        self.ax.clear()
        self.ax.imshow(self.image)
        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, ax=self.ax, with_labels=True, node_color='red', node_size=50)

    def save_graph(self, filename):
        nx.write_gml(self.graph, filename)

# Example usage
graph_builder = GraphBuilder('/Users/potosacho/Desktop/train.png')
plt.show()  # This will open a window to interact with the image
graph_builder.save_graph('data/train_lines.gml')  # Call this after closing the plot window

# Note: This code won't fully work in this notebook environment because it requires GUI interaction.
# You should run this code in a local Python environment where you can interact with the plot window.
