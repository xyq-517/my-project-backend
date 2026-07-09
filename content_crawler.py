# content_crawler.py - 公众号文章抓取模块
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import base64
import hashlib
import time
from datetime import datetime
import threading

# 配置文件路径
CONTENT_DB_PATH = r"D:\shengwu\using\unet-pytorch\content_db.json"

# 要抓取的关键词和公众号
KEYWORDS = ["肺结节", "肺癌筛查", "低剂量CT", "肺部健康"]

# 是否启用自动搜索（搜狗可能有反爬，建议先关闭）
ENABLE_AUTO_SEARCH = False

# 手动维护的精选文章列表（推荐方式，稳定可靠）
# 请将URL替换成真实的公众号文章链接
RECOMMENDED_ARTICLES = [
{
        "id": "wx_001",
        "title": "肺癌：了解它，远离它 | 肿瘤防治早知道",
        "url": "https://mp.weixin.qq.com/s/FdysMy6YBa6wrgz-3NUrig",
        "source": "健康中国",
        "coverImage": "",
        "publishDate": "2026-04-15",
        "category": "科普",
        "summary": "2026年4月15日—21日是第32个全国肿瘤防治宣传周，今年宣传周的主题为“早防早筛早治 同心携手抗癌”..."
    },
    {
        "id": "wx_002",
        "title": "胃肠镜检查是早期发现胃癌和结直肠癌的重要手段 | 肿瘤防治早知道",
        "url": "https://mp.weixin.qq.com/s/FTDvhz9ASeIPIwrxLDzuiA",
        "source": "健康中国",
        "coverImage": "",
        "publishDate": "2026-05-24",
        "category": "科普",
        "summary": "目前，胃癌和结直肠癌都是发病率较高的恶性肿瘤，而胃肠镜检查..."
    },
    {
        "id": "wx_003",
        "title": "令人闻风丧胆的癌症之王，居然跟这个坏习惯有关......",
        "url": "https://mp.weixin.qq.com/s/sTTDykFXo_xIc45Lwa2oVg",
        "source": "健康中国",
        "coverImage": "",
        "publishDate": "2025-11-20",
        "category": "科普",
        "summary": "每年 11 月的第三个周四是“世界胰腺癌日”，旨在提醒我们关注胰腺——这个深藏于腹腔之内的“沉默守护者”。"
    },
{
        "id": "wx_004",
        "title": "胸外科常见病知多少？分类 + 典型表现",
        "url": "https://mp.weixin.qq.com/s/YgVXYyeWqiO9_oztWDky3w",
        "source": "胸部肿瘤中心",
        "coverImage": "",
        "publishDate": "2026-01-08",
        "category": "科普",
        "summary": "在我们的身体中，胸腔是一个至关重要的“堡垒”，它保护着心脏..."
    },
{
        "id": "wx_005",
        "title": "16种常见急性腹痛的诊断与鉴别，值得你收藏",
        "url": "https://mp.weixin.qq.com/s/JzIzeiDANS6nodp5cgm-0A",
        "source": "急诊急救大平台",
        "coverImage": "",
        "publishDate": "2026-05-08",
        "category": "科普",
        "summary": "急腹症有很多原因，只有经过仔细采集病史、查体、合适的实验室和影像学检查后..."
    },
{
        "id": "wx_006",
        "title": "腹痛忍忍就好了？细数以腹痛症状为主的疾病→",
        "url": "https://mp.weixin.qq.com/s/mB13LeWZCZpzgQ8cX46-Kw",
        "source": "人卫健康",
        "coverImage": "",
        "publishDate": "2024-12-27",
        "category": "科普",
        "summary": "腹痛多数由腹部脏器疾病引起，但也可由腹腔外疾病及全身性疾病引起。腹痛的性质..."
    },

]


def load_content_db():
    """加载内容数据库"""
    if os.path.exists(CONTENT_DB_PATH):
        with open(CONTENT_DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"articles": [], "lastUpdate": ""}


def save_content_db(db):
    """保存内容数据库"""
    with open(CONTENT_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def download_image(url):
    """下载图片并转为base64"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://mp.weixin.qq.com/"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            # 获取图片格式
            content_type = resp.headers.get('Content-Type', 'image/jpeg')
            base64_data = base64.b64encode(resp.content).decode('utf-8')
            return f"data:{content_type};base64,{base64_data}"
    except Exception as e:
        print(f"[图片下载失败] {url}: {e}")
    return None


def clean_html_content(html_content):
    """清理HTML内容，适配rich-text"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # 移除脚本和样式
    for tag in soup(['script', 'style', 'iframe', 'video', 'audio']):
        tag.decompose()

    # 移除隐藏样式（关键修复）
    for tag in soup.find_all():
        if tag.get('style'):
            # 移除 visibility: hidden 和 opacity: 0
            style = tag['style']
            style = re.sub(r'visibility\s*:\s*hidden\s*;?', '', style)
            style = re.sub(r'opacity\s*:\s*0\s*;?', '', style)
            style = re.sub(r'opacity\s*:\s*0\.0+\s*;?', '', style)
            tag['style'] = style.strip()

    # 处理图片：下载并转为base64
    for img in soup.find_all('img'):
        src = img.get('data-src') or img.get('src', '')
        if src.startswith('http'):
            # 下载图片
            base64_img = download_image(src)
            if base64_img:
                img['src'] = base64_img
            else:
                # 下载失败，使用占位图或移除
                img.decompose()
                continue
        # 移除不必要的属性
        img.attrs = {'src': img.get('src', ''), 'style': 'max-width:100%;'}

    # 处理段落样式
    for p in soup.find_all('p'):
        p['style'] = 'margin:10px 0;line-height:1.6;'

    # 处理标题
    for h in soup.find_all(['h1', 'h2', 'h3']):
        h['style'] = 'margin:15px 0;font-weight:bold;'

    # 获取body内容或全部内容
    body = soup.find('body')
    if body:
        return str(body)
    return str(soup)


def fetch_wechat_article_content(url):
    """抓取公众号文章完整内容"""
    try:
        print(f"[抓取] 开始抓取: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = 'utf-8'
        print(f"[抓取] HTTP状态码: {resp.status_code}")

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 提取标题
        title = soup.find('h1', class_='rich_media_title')
        title = title.get_text(strip=True) if title else "无标题"
        print(f"[抓取] 文章标题: {title[:30]}...")

        # 提取正文内容
        content_div = soup.find('div', id='js_content')
        if not content_div:
            print(f"[抓取失败] 未找到内容区域 (id='js_content')")
            return None

        # 提取第一张图片作为封面
        cover_image = ""
        first_img = content_div.find('img')
        if first_img:
            img_src = first_img.get('data-src') or first_img.get('src', '')
            if img_src.startswith('http'):
                cover_image = img_src
                print(f"[抓取] 找到封面图: {img_src[:60]}...")

        print(f"[抓取] 找到内容区域，开始清理...")
        # 清理并处理内容
        clean_content = clean_html_content(str(content_div))
        print(f"[抓取] 内容清理完成，长度: {len(clean_content)}")

        return {
            "title": title,
            "content": clean_content,
            "coverImage": cover_image,
            "fetchTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"[文章抓取失败] {url}: {e}")
        import traceback
        traceback.print_exc()
        return None


def resolve_real_url(short_url):
    """将搜狗跳转链接解析为真实的微信文章URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # 如果是相对路径，加上搜狗域名
        if short_url.startswith('/'):
            short_url = 'https://weixin.sogou.com' + short_url

        # 方法1：尝试不跟随重定向，获取Location头
        resp = requests.get(short_url, headers=headers, timeout=15, allow_redirects=False)
        location = resp.headers.get('Location', '')
        if 'mp.weixin.qq.com' in location:
            return location

        # 方法2：跟随重定向，检查最终URL
        resp2 = requests.get(short_url, headers=headers, timeout=15, allow_redirects=True)
        if 'mp.weixin.qq.com' in resp2.url:
            return resp2.url

        # 方法3：从响应内容中提取跳转URL
        # 搜狗可能用 window.location 或 var url 跳转
        patterns = [
            r'window\.location\s*=\s*["\'](https?://mp\.weixin\.qq\.com/[^"\']+)["\']',
            r'var\s+url\s*=\s*["\'](https?://mp\.weixin\.qq\.com/[^"\']+)["\']',
            r'location\.href\s*=\s*["\'](https?://mp\.weixin\.qq\.com/[^"\']+)["\']',
            r'href\s*=\s*["\'](https?://mp\.weixin\.qq\.com/[^"\']+)["\']',
            r'(https?://mp\.weixin\.qq\.com/s/[^\s"\'<>]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, resp2.text)
            if match:
                return match.group(1)

        # 方法4：从Location头中提取（可能是相对路径）
        if location and not location.startswith('http'):
            # 可能是另一个搜狗跳转
            if 'mp.weixin' in location:
                return 'https://weixin.sogou.com' + location

        print(f"[URL解析] 未找到微信链接, 最终URL: {resp2.url[:100]}")
        return None

    except Exception as e:
        print(f"[URL解析失败] {short_url[:50]}...: {e}")
        return None


def auto_search_articles(keywords, max_per_keyword=3):
    """通过搜狗微信搜索自动发现新文章"""
    all_results = []

    # 更完整的请求头，模拟真实浏览器
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
        "Referer": "https://weixin.sogou.com/"
    }

    for keyword in keywords:
        try:
            # 使用更完整的搜索URL
            url = f"https://weixin.sogou.com/weixin?type=2&query={keyword}&ie=utf8&s_from=input&_sug_=n&_sug_type_="

            print(f"[搜索] 正在搜索 '{keyword}'...")
            resp = requests.get(url, headers=headers, timeout=15)
            resp.encoding = 'utf-8'

            # 检查是否被重定向到验证码页面
            if "验证" in resp.text or "captcha" in resp.text.lower():
                print(f"[搜索] '{keyword}' 触发验证码，跳过")
                time.sleep(5)
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # 尝试多种选择器
            items = soup.select('.news-list li') or soup.select('.news-box .news-list li') or soup.select(
                'ul.news-list li')

            if not items:
                # 打印部分HTML用于调试
                print(f"[搜索] '{keyword}' 未找到文章，可能页面结构变化")
                continue

            count = 0
            for item in items:
                if count >= max_per_keyword:
                    break

                # 尝试多种选择器获取标题和链接
                title_tag = item.select_one('.txt-info h3 a') or item.select_one('h3 a') or item.select_one(
                    'a[href*="mp.weixin"]')
                if not title_tag:
                    continue

                href = title_tag.get('href', '') or title_tag.get('data-share', '')
                if not href or 'mp.weixin' not in href:
                    # 尝试获取跳转链接
                    onclick = title_tag.get('onclick', '')
                    if 'location.href' in onclick:
                        import re
                        match = re.search(r"location\.href='([^']+)'", onclick)
                        if match:
                            href = match.group(1)

                if not href:
                    continue

                # 解析真实的微信文章URL
                real_url = resolve_real_url(href)
                if not real_url:
                    print(f"[跳过] {title_tag.get_text(strip=True)[:20]}... - 无法获取真实URL")
                    continue

                title = title_tag.get_text(strip=True)
                source_tag = item.select_one('.account') or item.select_one('.s-p')
                source = source_tag.get_text(strip=True).replace("微信号：", "").replace("来源：",
                                                                                        "") if source_tag else "微信公众号"
                summary_tag = item.select_one('.txt-info p') or item.select_one('p')
                summary = summary_tag.get_text(strip=True)[:100] if summary_tag else ""
                date_tag = item.select_one('.s2') or item.select_one('.time')
                publish_date = date_tag.get_text(strip=True) if date_tag else ""

                # 生成唯一ID（用真实URL生成，避免重复）
                article_id = "auto_" + hashlib.md5(real_url.encode()).hexdigest()[:10]

                all_results.append({
                    "id": article_id,
                    "title": title,
                    "url": real_url,
                    "source": source,
                    "coverImage": "",
                    "publishDate": publish_date,
                    "category": "科普",
                    "summary": summary,
                    "fullContent": "",
                    "isAuto": True
                })
                count += 1

            print(f"[搜索] '{keyword}' 找到 {count} 篇文章")

        except Exception as e:
            print(f"[搜索] '{keyword}' 搜索失败: {e}")

        # 随机延迟，避免触发反爬
        time.sleep(3 + (hash(keyword) % 3))

    return all_results


def crawl_all_articles():
    """抓取所有推荐文章的完整内容"""
    # 只使用手动推荐文章，清空旧数据
    db = {"articles": [], "lastUpdate": ""}

    print(f"[{datetime.now()}] 开始抓取文章...")
    print("[搜索] 自动搜索已禁用，使用手动推荐文章")

    # ====== 只抓取手动推荐文章 ======
    for article in RECOMMENDED_ARTICLES:

        print(f"[抓取] {article['title']}...")

        # 抓取完整内容
        content_data = fetch_wechat_article_content(article["url"])

        if content_data:
            article_data = {
                **article,
                "fullContent": content_data["content"],
                "fetchTime": content_data["fetchTime"]
            }
            # 如果文章没有封面图，使用抓取到的第一张图片
            if not article.get("coverImage") and content_data.get("coverImage"):
                article_data["coverImage"] = content_data["coverImage"]

            db["articles"].append(article_data)
            print(f"[成功] {article['title']}")
        else:
            # 抓取失败，保存基础信息
            db["articles"].append(article)
            print(f"[失败] {article['title']}")

        # 避免请求过快
        time.sleep(2)

    db["lastUpdate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_content_db(db)

    print(f"[{datetime.now()}] 抓取完成，共 {len(db['articles'])} 篇文章")
    return db


def start_auto_refresh(interval_hours=24):
    """启动定时刷新任务"""

    def refresh_loop():
        while True:
            try:
                crawl_all_articles()
            except Exception as e:
                print(f"[定时任务错误] {e}")

            # 等待指定时间
            time.sleep(interval_hours * 3600)

    # 启动后台线程
    thread = threading.Thread(target=refresh_loop, daemon=True)
    thread.start()
    print(f"[定时任务] 已启动，每 {interval_hours} 小时刷新一次")


# 如果直接运行此文件，执行抓取
if __name__ == "__main__":
    crawl_all_articles()
