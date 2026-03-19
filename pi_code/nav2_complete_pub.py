#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from action_msgs.msg import GoalStatusArray

class NavCompletePublisher(Node):
    def __init__(self):
        super().__init__('nav_complete_publisher')
        
        # 1. 订阅Nav2的目标状态话题（检测导航完成）
        self.nav_status_sub = self.create_subscription(
            GoalStatusArray,
            '/navigation2/goal_status',  # Nav2状态话题
            self.nav_status_callback,
            10)
        
        # 2. 发布“导航完成”话题给树莓派（自定义话题名）
        self.raspi_pub = self.create_publisher(
            String,
            '/robot/nav_complete',  # 跨设备话题名
            10)
        
        self.nav_completed = False  # 标记导航是否完成

    def nav_status_callback(self, msg):
        """回调函数：检测Nav2导航是否完成"""
        # Nav2的GoalStatus中，状态码4=SUCCEEDED（导航完成）
        for status in msg.status_list:
            if status.status == 4 and not self.nav_completed:
                self.nav_completed = True
                self.get_logger().info("✅ 导航任务完成！")
                
                # 发布消息给树莓派
                send_msg = String()
                send_msg.data = "nav_complete"  # 自定义消息内容（可改）
                self.raspi_pub.publish(send_msg)
                self.get_logger().info("📤 已向树莓派发送导航完成消息")
                
                # 重置标记（可选：若需多次导航）
                # self.nav_completed = False

def main(args=None):
    rclpy.init(args=args)
    node = NavCompletePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
