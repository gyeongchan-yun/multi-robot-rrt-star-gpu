"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import os
import sys
import time

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

import threading
from queue import Queue

show_animation = True

# -- Global variables for multi-robot -- #
robot_positions = []
lock = threading.Lock()
ROBOT_RADIUS = 0.5
# -------------------------------------- #

def log_fn(thd_name, msg):
    print('[INFO][{}] {}'.format(thd_name, msg))


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=50.0
                 ):
        super().__init__(start, goal, obstacle_list,
                         rand_area, expand_dis, path_resolution, goal_sample_rate, max_iter)
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.static_obstacle_list = obstacle_list

    def planning(self, ret_queue, animation=True, search_until_max_iter=True):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_max_iter: search until max iteration for path improving or not
        """
        global robot_positions
        thd_name = threading.currentThread().getName()
        thd_idx = int(thd_name.split('-')[-1])
        log_fn(thd_name, "thread name: {}".format(thd_name))
        self.node_list = [self.start]
        """
        lock.acquire()
        log_fn(thd_name, "start point ==> x:{}, y:{}".format(self.start.x, self.start.y))
        robot_position = (self.start.x, self.start.y, ROBOT_RADIUS)
        if robot_position not in robot_positions:
            robot_positions.append(robot_position)
        print(robot_positions)
        lock.release()
        """
        for i in range(self.max_iter):
            log_fn(thd_name, "Iter: {}, number of nodes: {}".format(i, len(self.node_list)))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            lock.acquire()
            other_robot_positions = [(position) for i, position in enumerate(robot_positions) if i != thd_idx and len(position) > 0]
            if other_robot_positions:
                other_robot_positions = other_robot_positions[0]
            self.obstacle_list = self.obstacle_list + other_robot_positions
            # print("self.obstacle_list", self.obstacle_list)
            lock.release()
            if self.check_collision(new_node, self.obstacle_list):
                # log_fn(thd_name, "collision free!")
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    lock.acquire()
                    new_robot_position = (new_node.x, new_node.y, ROBOT_RADIUS)
                    if new_robot_position not in robot_positions[thd_idx]:
                        robot_positions[thd_idx].append(new_robot_position)
                    lock.release()
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            # if animation and i % 5 == 0:
            #    self.draw_graph(rnd)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    ret_queue.put(self.generate_final_course(last_index))
                    return self.generate_final_course(last_index)

        log_fn(thd_name, "reached max iteration")
        log_fn(thd_name, "length of node_list: {}".format(len(self.node_list)))
        last_index = self.search_best_goal_node()
        if last_index:
            ret_queue.put(self.generate_final_course(last_index))
            return self.generate_final_course(last_index)

        ret_queue.put(None)
        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def draw_graph_multi_robot(rrt_star_robots, path_list):
    plt.clf()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    color_list3 = ["#feb24c", "#377eb8","#e41a1c"]
    for robot, path, color in zip(rrt_star_robots, path_list, color_list3):
        """
        for node in robot.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")
        """
        for (ox, oy, size) in robot.obstacle_list:
            if size < 0.5:
                robot.plot_circle(ox, oy, size, color=color)
            robot.plot_circle(ox, oy, size)
        #for (ox, oy, size) in robot.static_obstacle_list:
        #    robot.plot_circle(ox, oy, size)

        plt.plot(robot.start.x, robot.start.y, "xr")
        plt.plot(robot.end.x, robot.end.y, "xr")
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)
    plt.show()

def main():
    global robot_positions

    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]  # [x,y,size(radius)]
    space_size = [-2, 15]

    # Set Initial parameters
    #"""                        # start   goal
    multi_robot_config_list = [
                                [[0, 0], [6, 10]],
                                [[12, 0], [7, 8]],
                              ]
    num_robots = 2
    #"""
    """
    multi_robot_config_list = [
                                [[0, 0], [6, 10]],
                              ]
    num_robots = 1
    """
    if len(multi_robot_config_list) != num_robots:
        print("The configuration of multi-robot is not enough!")
        return

    for _ in range(0, num_robots):
        robot_positions.append([])

    rrt_star_robots = []
    for i in range(0, num_robots):
        rrt_star_robots.append(RRTStar(start=multi_robot_config_list[i][0],
                                        goal=multi_robot_config_list[i][1],
                                        rand_area=space_size,
                                        obstacle_list=obstacle_list,
                                        max_iter=50))

    start_time = time.time()

    thd_ret_queue = Queue()
    multi_robots = [threading.Thread(target=rrt_star_robots[i].planning,
                                     name='robot-{}'.format(i),
                                     args=(thd_ret_queue, show_animation,)
                                     ) for i in range(0, num_robots)]
    for robot in multi_robots:
        robot.start()
    for robot in multi_robots:
        robot.join()

    print("[INFO] Execution time: {0:.2f} secs".format(time.time() - start_time))

    path_list = []
    while not thd_ret_queue.empty():
        path = thd_ret_queue.get()
        if path is None:
            print("Cannot find path")
            path = []
        else:
            print("found path!!")
        path_list.append(path)
    if not path_list:
        return
    print(path_list)
    # Draw final path
    if show_animation:
        draw_graph_multi_robot(rrt_star_robots, path_list)
        """
        for i in range(0, num_robots):
            rrt_star_robots[i].draw_graph()
            if path_list[i]:
                plt.plot([x for (x, y) in path_list[i]], [y for (x, y) in path_list[i]], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
        plt.show()
        """

if __name__ == '__main__':
    main()
