# baglab-mcap-backend

ROS 2 依存なしで MCAP ファイルを高速に読み出す baglab の第3バックエンド。

## 動機

baglab には既存の2つのバックエンドがある:

| バックエンド | 速度 | ROS 2 依存 | 対応フォーマット |
|---|---|---|---|
| rosbags (Python) | 遅い | なし | mcap, db3 |
| rosbag2_cpp (C++) | 速い | **あり** | mcap, db3 |

rosbag2_cpp は高速だが ROS 2 環境が必須であり、`source` せずに実行するとパースに失敗する。
また一括ロードで OOM、マテリアライズに5分以上かかるなどの課題がある。

ROS 2 Jazzy (Iron) 以降、デフォルトの bag フォーマットが db3 から **mcap** に変わったことで、
mcap 特化のバックエンドを作る価値が高まった。

## 技術選定

### アプローチ: mcap C++ reader + 自前 CDR デシリアライザ

MCAP ファイルには ros2msg 形式のスキーマが埋め込まれており、ROS 2 の rosidl 型情報なしでメッセージ構造を把握できる。
CDR (Common Data Representation) は OMG 標準 ([formal/02-06-51](https://www.omg.org/cgi-bin/doc?formal/02-06-51)) であり ROS 2 固有ではない。
これら2点を組み合わせることで ROS 2 ランタイム依存を完全に排除した。

- **I/O**: [foxglove/mcap](https://github.com/foxglove/mcap) C++ ヘッダオンリーライブラリ + mmap
- **スキーマ解析**: MCAP 内蔵の ros2msg テキストをパースしてメッセージレイアウトを構築
- **CDR デシリアライズ**: バイナリストリームをスキーマに従って直接 columnar numpy 配列に展開
- **Python バインディング**: pybind11

### なぜこのアプローチか — 実験による根拠

#### 実験1: mcap Python ライブラリ vs rosbags (テストbag 140KB, 992 msgs)

mcap ライブラリの raw I/O は rosbags より高速だが、mcap-ros2-support のデシリアライズは遅い:

| 手法 | 平均時間 | vs rosbags |
|---|---|---|
| rosbags (AnyReader + get_dataframe) | 15.1ms | baseline |
| mcap-ros2-support (deserialize + DataFrame) | 22.7ms | 0.67x (遅い) |
| **hybrid (mcap raw + rosbags CDR deser)** | **5.9ms** | **2.6x** |

mcap-ros2-support が遅い原因は動的型生成のオーバーヘッド。hybrid 手法が最も高速。

#### 実験2: ボトルネック分析 (テストbag, /test/twist 496 msgs)

hybrid 手法の処理時間内訳:

| 処理 | 時間 | 割合 |
|---|---|---|
| mcap Python I/O | 3.27ms | 62.7% |
| CDR デシリアライズ (rosbags) | 1.47ms | 28.2% |
| フィールド抽出 | 0.20ms | 3.8% |
| DataFrame 構築 | 0.28ms | 5.3% |

最大ボトルネックは **mcap Python I/O (63%)** であり、CDR デシリアライズ単体の C 化では効果が限定的。
mcap I/O ごと C++ 化する必要がある。

#### 実験3: C++ 化の効果 (テストbag, 全トピック → DataFrame)

| 手法 | 平均時間 | vs rosbags |
|---|---|---|
| rosbags | 16.1ms | 1.0x |
| baglab_mcap_backend (ifstream) | 1.3ms | 12x |
| **baglab_mcap_backend (mmap)** | **0.3ms** | **53x** |

#### 実験4: 実データ (19GB mcap, 635 topics, 912K msgs, 60s recording)

`/sensing/imu/gyro_bias` (12,000 msgs, Vector3Stamped) の読み出し:

| 手法 | 時間 | vs rosbags |
|---|---|---|
| rosbags | 2,539ms | 1.0x |
| Python mcap raw read (no deser) | 3,153ms | — |
| baglab_mcap_backend (ifstream) | 2,157ms | 1.2x |
| **baglab_mcap_backend (mmap)** | **324ms** | **7.8x** |

mmap 化により ifstream 比で **6.6倍高速化**。19GB ファイルでの rosbags 比 **7.8倍**を達成。

### mmap が効果的な理由

19GB ファイルは 12,662 チャンクに分割されている。1トピックの読み出しでも対象チャンクが全体の約50%に分散しており、
大量の seek + read が発生する。ifstream ではユーザ空間バッファリングのオーバーヘッドが大きいが、
mmap ではカーネルのページキャッシュに直接アクセスするため seek コストがほぼゼロになる。

## アーキテクチャ

```
baglab_mcap_backend/
  src/
    cpp/
      mcap_reader.{hpp,cpp}       # MCAP I/O (foxglove C++ reader + mmap)
      schema_parser.{hpp,cpp}     # ros2msg スキーマ → MessageLayout ツリー
      cdr_deserializer.{hpp,cpp}  # CDR バイナリ → columnar numpy 配列
      bindings.cpp                # pybind11 (get_topics, read_topic)
    baglab_mcap_backend/
      __init__.py                 # Python パッケージ
  third_party/
    mcap_repo/                    # foxglove/mcap (git clone)
  CMakeLists.txt                  # pybind11 + lz4 + zstd
  pyproject.toml                  # scikit-build-core
```

### 処理フロー

```
MCAP ファイル
  ↓ mmap (zero-copy)
foxglove McapReader
  ↓ readSummary() → トピック一覧 + スキーマ取得
  ↓ readMessages(topicFilter) → チャンク単位でフィルタリング
schema_parser
  ↓ ros2msg テキスト → MessageLayout (フィールド型・ネスト構造)
ExtractionPlan
  ↓ MessageLayout + 対象フィールド → Step列 (READ_SCALAR / SKIP_SCALAR / ...)
CdrReader + execute_plan
  ↓ CDR バイナリを Step に従って走査、対象フィールドのみ Column に蓄積
build_columnar_result
  ↓ Column → numpy 配列 (memcpy)
Python dict {"__timestamps__": ndarray, "field.name": ndarray, ...}
  ↓ (baglab 側)
pandas DataFrame
```

## ROS 2 仕様との互換性

### 対応状況

| ROS 2 ディストロ | MCAP format | CDR | スキーマ | 対応 |
|---|---|---|---|---|
| Humble (2022) | v0 | XCDR1 | ros2msg | OK |
| Iron (2023) | v0 | XCDR1 | ros2msg | OK |
| Jazzy (2024) | v0 | XCDR1 | ros2msg | OK |
| Kilted (2025) | v0 | XCDR1 | ros2msg | OK |

### CDR エンコーディング

ROS 2 は全ディストロ (Humble〜Kilted) で **XCDR1** を使用。
`rmw_fastrtps` は [PR #746](https://github.com/ros2/rmw_fastrtps/pull/746) で Fast CDR v2 対応した際に
`CdrVersion::XCDRv1` を明示的にハードコードしている。

XCDR2 への移行について eProsima は [Issue #811](https://github.com/ros2/rmw_fastrtps/issues/811) にて
「ロードマップにない」と明言 (2025年3月)。

本バックエンドは encapsulation ID を検証し、XCDR2 (0x0006〜0x000b) を検出した場合は
明示的なエラーメッセージで安全に失敗する。

### スキーマエンコーディング

`ros2msg` のみ対応。`ros2idl` エンコーディング (`.idl` 定義のみのカスタムメッセージ) は
エラーメッセージで rosbags バックエンドへの切り替えを案内する。
標準 ROS 2 メッセージは全て `ros2msg` で提供されるため、実用上の問題はない。

## 制限事項

- **mcap 形式のみ対応**: db3 (sqlite3) 形式は読めない。db3 は rosbags バックエンドを使用すること
- **ros2idl スキーマ非対応**: `.idl` 定義のみのカスタムメッセージは読めない
- **XCDR2 非対応**: 将来の ROS 2 が CDR フォーマットを変更した場合は更新が必要
- **big-endian 未テスト**: CDR_BE (0x0000) のコードパスは実装済みだが x86 環境でのテストのみ

## ビルド

```bash
# 依存: pybind11, lz4, zstd
sudo apt install liblz4-dev libzstd-dev

# インストール
cd baglab_mcap_backend
pip install .
```

## 使い方

```python
import baglab

# backend="mcap" を指定 (auto でも mcap が優先される)
bag = baglab.load("path/to/rosbag_dir", backend="mcap")
df = bag["/topic_name"]
```
