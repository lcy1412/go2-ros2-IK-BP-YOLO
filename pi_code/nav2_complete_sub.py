#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class RaspiSubscriber(Node):
    def __init__(self):
        super().__init__('raspi_subscriber')
        
        # 订阅机器狗的导航完成话题
        self.subscription = self.create_subscription(
            String,
            '/robot/nav_complete',  # 和机器狗的话题名一致
            self.listener_callback,
            10)
        self.subscription  # 防止未使用变量警告

    def listener_callback(self, msg):
        """收到消息后的回调：执行树莓派的动作"""
        self.get_logger().info(f"📥 收到机器狗消息：{msg.data}")
        
        # ========== 这里写树莓派的动作逻辑 ==========
        # 示例1：打印提示
        print("机器狗导航完成！执行树莓派动作...")
        

def main(args=None):
    rclpy.init(args=args)
    node = RaspiSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
