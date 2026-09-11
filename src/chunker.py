from langchain_text_splitters import MarkdownHeaderTextSplitter


def split_prose_and_code(markdown_text: str, base_metadata: dict) -> list[dict]:
    chunks = []

    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    md_header_splits = markdown_splitter.split_text(markdown_text)

    for split in md_header_splits:
        content = split.page_content.strip()
        if content:
            chunks.append(
                {
                    "content": content,
                    "metadata": {
                        **base_metadata,
                        **split.metadata,
                        "type": "mixed_context",
                    },
                }
            )

    return chunks
