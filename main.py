import sys
from src.automator import GameAutomator

if __name__ == "__main__":
    print("======= 注意 =======")
    print("请确保游戏在16:9下运行，本程序不支持其它比例")
    print("程序会产生大量拖拽动作，建议全屏模式下运行，或让游戏窗口尽量大，覆盖你的桌面文件")
    print("请右键选择管理员模式运行此程序，否则无法操作NIKKE")
    print("如果需要停止，将鼠标快速向左上角滑动")
    print("请先进入一次小游戏，暂停后【快速退出】，结算当次游戏后再执行程序")
    print("======= 注意 =======")
    print("请选择游戏模式:")
    print("1. 10x16 (默认)")
    print("2. 9x15")
    choice = input("请输入序号 (1/2): ").strip()
    
    if choice not in ["1", "2"]:
        print("输入无效，默认使用模式 1")
        choice = "1"

    loop_input = input("是否启用循环模式 (y/n)? [默认: n]: ").strip().lower()
    enable_loop = (loop_input == 'y')
        
    bot = GameAutomator(choice, enable_loop)
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n程序被用户强制停止。")
    except Exception as e:
        print(f"\n程序发生错误: {e}")
        input("按回车键退出...")