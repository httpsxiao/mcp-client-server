import asyncio
import os
import sys
import json
import re
from typing import Optional
from contextlib import AsyncExitStack

# LLM 接入
from openai import OpenAI
from dotenv import load_dotenv

# mcp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 加载 .env 文件，确保 API Key 受到保护
load_dotenv()


class MCPClient:
    def __init__(self):
        """
        初始化 MCP 客户端
        """
        # 上下文管理器
        self.exit_stack = AsyncExitStack()

        # 读取 BASE URL
        self.base_url = os.getenv("BASE_URL")

        # 读取大模型 API Key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # 读取大模型
        self.model = os.getenv("MODEL")

        if not self.openai_api_key:
            raise ValueError(
                "❌ 未找到 OpenAI API Key，请在 .env 文件中设置OPENAI_API_KEY"
            )

        # 初始化大模型
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)

        # 与 MCP Server 会话 session
        self.session: Optional[ClientSession] = None

    async def connect_to_server(self, server_script_path: str):
        """
        连接到MCP服务器并列出可用工具
        """

        # Server 必须是 python 或者 js
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是.py或.js文件")

        # 拼接 MCP 服务器参数
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path]
        )

        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport

        # MCP 会话管理
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()

        # 列出可用的 MCP 工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器,支持以下工具:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """
        使用大模型处理查询并调用可用的 MCP 工具
        """

        # 将用户输入 + 可用的工具给到大模型
        messages = [
            {"role": "user", "content": query},
        ]
        toolList = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                },
            }
            for tool in toolList.tools
        ]

        # 上述内容作为参数开启大模型会话
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools,
        )

        # 获得大模型返回的内容
        modelResult = response.choices[0]

        if modelResult.finish_reason == "tool_calls":
            # 如何是需要使用工具，就解析出来需要的 工具 + 参数
            tool_call = modelResult.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # 执行工具
            toolResult = await self.session.call_tool(tool_name, tool_args)
            print(
                f"\n\n[Callingtool {tool_name} with args {tool_args}]\n toolResult is {toolResult.content[0].text}\n"
            )

            # 将模型返回的调用哪个工具数据和工具执行完成后的数据都存入 messages 中，以 tool 身份发送
            messages.append(modelResult.message.model_dump())
            messages.append(
                {
                    "role": "tool",
                    "content": toolResult.content[0].text,
                    "tool_call_id": tool_call.id,
                }
            )

            # 将工具的结果再返回给大模型用于生产最终的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            # 过滤推理模型的推理部分，也可以在上一步设置 max_tokens 参数，但是文字可能会被截断
            rmThinkContent = re.sub(
                r"<think>.*?</think>",
                "",
                response.choices[0].message.content,
                flags=re.DOTALL,
            )

            # 最终大模型二次的返回结果给到用户
            return rmThinkContent

        return modelResult.message.content

    async def chat_loop(self):
        """
        运行交互式聊天循环
        """
        # 给用户的提示
        print("\nMCP 客户端已启动！输入 'quit' 退出")

        while True:
            try:
                query = input("\n请输入内容:").strip()

                # 如果输入 quit，则停止运行
                if query.lower() == "quit":
                    break

                # 用户的输入给到大模型
                response = await self.process_query(query)

                # 输出大模型的结果
                print(f"\n🤖 大模型: {response}")
            except Exception as e:
                print(f"\n⚠️ 发生错误: {str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("python client.py<path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
