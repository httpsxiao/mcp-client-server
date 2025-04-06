import asyncio
import json
import httpx
import random
from typing import Any
from mcp.server.fastmcp import FastMCP

# 初始化MCP服务器
mcp = FastMCP("WeatherServer")
random_array = ["晴", "阴", "雨", "雪", "雾", "雷", "风", "沙", "尘", "霾"]


async def mock_fetch_weather(city: str) -> str | None:
    """
    获取指定城市的天气信息
    """
    temperature = random.randrange(5, 30)
    humidity = random.randrange(30, 70)

    await asyncio.sleep(1)

    return f"🏙 {city} 的天气：🌡️ {temperature}℃，💧 {humidity}%，🌬️ {random.choice(random_array)}"


@mcp.tool()
async def query_weather(city: str) -> str:
    """
    输入指定城市的英文名称，返回今日天气查询结果。
    :param city: 城市名称（需使用英文）
    :return: 天气信息
    """
    result = await mock_fetch_weather(city)

    return result


if __name__ == "__main__":
    # 以标准I/O方式运行MCP服务器
    mcp.run(transport="stdio")
