import { useMemo, useState } from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import EnhancedCharts from "@/components/EnhancedCharts"
import "./App.css"

type UploadFile = {
  name: string
  type: string
  size: number
  base64: string
}

type Section = {
  title: string
  excerpt: string
  summary: string
  thinking: string
  keywords: { term: string; score: number }[]
}

type DocumentResult = {
  name: string
  word_count: number
  summary: string
  conclusions: string[]
  keywords: { term: string; score: number }[]
  sections: Section[]
}

type ConflictItem = {
  topic: string
  doc_a: string
  doc_b: string
  sentiment_a: number
  sentiment_b: number
}

type AnalysisResult = {
  documents: DocumentResult[]
  keyword_chart: { name: string; value: number }[]
  conclusion_graph: { nodes: string[]; edges: { source: string; target: string; weight: number }[] }
  comparison: {
    similarity: number[][]
    overlap_keywords: string[]
    unique_keywords: Record<string, string[]>
  }
  conflicts: ConflictItem[]
  // 新增字段
  document_type: string
  recommended_charts: string[]
  enhanced_charts: {
    wordcloud: Array<{ text: string; value: number }>
    pie: Array<{ name: string; value: number }>
    timeline: Array<{ date: string; event: string; importance: number }>
    entity_network: {
      nodes: Array<{ id: string; group: number }>
      links: Array<{ source: string; target: string; value: number }>
    }
  }
}

function App() {
  const [files, setFiles] = useState<UploadFile[]>([])

  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + file.size, 0),
    [files]
  )

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [notes, setNotes] = useState("")

  const overview = useMemo(() => {
    if (!result) return null
    const docCount = result.documents.length
    const totalWords = result.documents.reduce(
      (sum, doc) => sum + doc.word_count,
      0
    )
    const avgWords = docCount ? Math.round(totalWords / docCount) : 0
    const sectionCount = result.documents.reduce(
      (sum, doc) => sum + doc.sections.length,
      0
    )
    const topKeywords = result.keyword_chart.slice(0, 12)
    const overlapCount = result.comparison.overlap_keywords.length
    const conflictCount = result.conflicts.length
    return {
      docCount,
      totalWords,
      avgWords,
      sectionCount,
      topKeywords,
      overlapCount,
      conflictCount,
    }
  }, [result])

  const handleFiles = async (fileList: FileList | null) => {
    if (!fileList) return
    setError(null)
    const incoming = Array.from(fileList)

    const encoded = await Promise.all(
      incoming.map(async (file) => {
        const buffer = await file.arrayBuffer()
        const binary = new Uint8Array(buffer)
        let binaryString = ""
        binary.forEach((byte) => {
          binaryString += String.fromCharCode(byte)
        })
        const base64 = btoa(binaryString)
        return {
          name: file.name,
          type: file.type || "application/octet-stream",
          size: file.size,
          base64,
        }
      })
    )

    setFiles((prev) => [...prev, ...encoded])
  }

  const removeFile = (name: string) => {
    setFiles((prev) => prev.filter((file) => file.name !== name))
  }

  const analyze = async () => {
    if (!files.length) {
      setError("请先上传 Word / PDF / 文本文件")
      return
    }
    setLoading(true)
    setError(null)
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ files, notes }),
      })
      if (!response.ok) {
        throw new Error("分析失败，请稍后再试")
      }
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "分析失败")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-10">
        <header className="flex flex-col gap-4">
          <Badge className="w-fit bg-indigo-500/20 text-indigo-200">AI 文档平台</Badge>
          <h1 className="text-3xl font-semibold tracking-tight md:text-4xl">
            Word / PDF / 论文 → 结构化知识 + 图表 + 结论
          </h1>
          <p className="text-slate-300">
            学生论文、研究报告、行业白皮书统一分析。自动输出研究结论图、文献对比和观点冲突检测。
          </p>
        </header>

        <div className="grid gap-6 lg:grid-cols-[360px_1fr]">
          <Card className="border-slate-800 bg-slate-900/60">
            <CardHeader>
              <CardTitle>上传文档</CardTitle>
              <CardDescription>支持 PDF、Word、纯文本。可多文件上传用于对比。</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-xl border border-dashed border-slate-700 bg-slate-950/40 p-4">
                <Input
                  type="file"
                  multiple
                  accept=".pdf,.doc,.docx,.txt"
                  onChange={(event) => handleFiles(event.target.files)}
                />
                <p className="mt-2 text-xs text-slate-400">拖拽文件到此处或直接选择。</p>
              </div>
              <div className="space-y-2">
                {files.map((file) => (
                  <div
                    key={file.name}
                    className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2 text-sm"
                  >
                    <div>
                      <p className="font-medium text-slate-100">{file.name}</p>
                      <p className="text-xs text-slate-400">
                        {(file.size / 1024).toFixed(1)} KB
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-slate-300 hover:text-white"
                      onClick={() => removeFile(file.name)}
                    >
                      移除
                    </Button>
                  </div>
                ))}
                {!files.length && (
                  <p className="text-sm text-slate-500">暂未添加文件</p>
                )}
              </div>
              <div className="space-y-2">
                <p className="text-sm text-slate-300">分析备注（可选）</p>
                <Textarea
                  placeholder="例如：关注结论章节，输出图表摘要"
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>已上传 {files.length} 份文档</span>
                  <span>总大小 {(totalSize / 1024).toFixed(1)} KB</span>
                </div>
                {loading && <Progress value={70} />}
                {error && <p className="text-sm text-rose-300">{error}</p>}
              </div>
              <Button
                className="w-full bg-indigo-500 text-white hover:bg-indigo-400"
                onClick={analyze}
                disabled={loading}
              >
                {loading ? "正在分析..." : "开始分析"}
              </Button>
            </CardContent>
          </Card>

          <Card className="border-slate-800 bg-slate-900/60">
            <CardHeader>
              <CardTitle>分析结果</CardTitle>
              <CardDescription>输出结构化知识、图表与研究结论图。</CardDescription>
            </CardHeader>
            <CardContent>
              {!result ? (
                <div className="flex h-[420px] flex-col items-center justify-center gap-3 text-center text-slate-400">
                  <p>上传文档后即可查看分析结果</p>
                  <p className="text-xs">支持研究结论图、文献对比、观点冲突检测。</p>
                </div>
              ) : (
                <Tabs defaultValue="overview" className="space-y-4">
                  <TabsList className="flex flex-wrap gap-2 bg-slate-950/70">
                    <TabsTrigger value="overview">总览</TabsTrigger>
                    <TabsTrigger value="summary">摘要</TabsTrigger>
                    <TabsTrigger value="conclusions">结论</TabsTrigger>
                    <TabsTrigger value="sections">章节</TabsTrigger>
                    <TabsTrigger value="keywords">关键词</TabsTrigger>
                    <TabsTrigger value="charts">图表</TabsTrigger>
                    <TabsTrigger value="comparison">文献对比</TabsTrigger>
                    <TabsTrigger value="conflict">观点冲突</TabsTrigger>
                  </TabsList>

                  <TabsContent value="overview" className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <Card className="border-slate-800 bg-slate-950/40">
                        <CardHeader>
                          <CardTitle className="text-base">文档规模</CardTitle>
                          <CardDescription>整体规模与章节统计</CardDescription>
                        </CardHeader>
                        <CardContent className="grid gap-3 text-sm text-slate-300">
                          <div className="flex items-center justify-between">
                            <span>文档数量</span>
                            <span className="text-slate-100">{overview?.docCount ?? 0}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span>总字数</span>
                            <span className="text-slate-100">{overview?.totalWords ?? 0}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span>平均字数</span>
                            <span className="text-slate-100">{overview?.avgWords ?? 0}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span>章节数量</span>
                            <span className="text-slate-100">{overview?.sectionCount ?? 0}</span>
                          </div>
                        </CardContent>
                      </Card>
                      <Card className="border-slate-800 bg-slate-950/40">
                        <CardHeader>
                          <CardTitle className="text-base">分析备注</CardTitle>
                          <CardDescription>用户输入的关注点</CardDescription>
                        </CardHeader>
                        <CardContent className="text-sm text-slate-300">
                          {notes.trim() ? notes : "未填写备注"}
                        </CardContent>
                      </Card>
                    </div>
                    <div className="grid gap-4 md:grid-cols-2">
                      <Card className="border-slate-800 bg-slate-950/40">
                        <CardHeader>
                          <CardTitle className="text-base">关键词聚焦</CardTitle>
                          <CardDescription>高频关键词与研究焦点</CardDescription>
                        </CardHeader>
                        <CardContent className="flex flex-wrap gap-2">
                          {overview?.topKeywords.map((item) => (
                            <Badge key={item.name} className="bg-indigo-500/20 text-indigo-200">
                              {item.name}
                            </Badge>
                          ))}
                        </CardContent>
                      </Card>
                      <Card className="border-slate-800 bg-slate-950/40">
                        <CardHeader>
                          <CardTitle className="text-base">风险提示</CardTitle>
                          <CardDescription>差异与冲突快速预警</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-2 text-sm text-slate-300">
                          <div className="flex items-center justify-between">
                            <span>重叠关键词</span>
                            <span className="text-slate-100">{overview?.overlapCount ?? 0}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span>冲突条目</span>
                            <span className="text-slate-100">{overview?.conflictCount ?? 0}</span>
                          </div>
                          <p className="text-xs text-slate-500">
                            冲突条目来源于情感倾向差异，建议重点核对。
                          </p>
                        </CardContent>
                      </Card>
                    </div>
                  </TabsContent>

                  <TabsContent value="summary" className="space-y-6">
                    {result.documents.map((doc) => (
                      <Card key={doc.name} className="border-slate-800 bg-slate-950/40">
                        <CardHeader>
                          <div className="flex flex-wrap items-center gap-2">
                            <CardTitle className="text-base">{doc.name}</CardTitle>
                            <Badge className="bg-slate-800 text-slate-200">
                              {doc.word_count} 字
                            </Badge>
                          </div>
                          <CardDescription>摘要与关键词速览</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-4">
                          <p className="text-sm text-slate-300">{doc.summary}</p>
                          <div className="flex flex-wrap gap-2">
                            {doc.keywords.map((item) => (
                              <Badge key={item.term} className="bg-slate-800 text-slate-200">
                                {item.term}
                              </Badge>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  <TabsContent value="conclusions" className="space-y-6">
                    {result.documents.map((doc) => (
                      <Card key={doc.name} className="border-slate-800 bg-slate-950/40">
                        <CardHeader>
                          <CardTitle className="text-base">{doc.name}</CardTitle>
                          <CardDescription>研究结论与核心发现</CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-2 text-sm text-slate-300">
                          {doc.conclusions.map((item, index) => (
                            <p key={index}>• {item}</p>
                          ))}
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  <TabsContent value="sections" className="space-y-6">
                    {result.documents.map((doc) => (
                      <Card key={doc.name} className="border-slate-800 bg-slate-950/40">
                        <CardHeader>
                          <CardTitle className="text-base">{doc.name} - 章节结构</CardTitle>
                          <CardDescription>分结构拆解 + 思路分析</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <div className="grid gap-4 lg:grid-cols-[220px_1fr]">
                            <div className="space-y-3 text-sm text-slate-300">
                              <p className="text-xs text-slate-500">结构目录</p>
                              {doc.sections.map((section, index) => (
                                <div key={section.title} className="flex items-center gap-2">
                                  <Badge className="bg-slate-800 text-slate-200">{index + 1}</Badge>
                                  <span>{section.title}</span>
                                </div>
                              ))}
                            </div>
                            <div className="space-y-4">
                              {doc.sections.map((section) => (
                                <Card key={section.title} className="border-slate-800 bg-slate-900/40">
                                  <CardHeader>
                                    <CardTitle className="text-sm">{section.title}</CardTitle>
                                    <CardDescription>章节摘要与关键词</CardDescription>
                                  </CardHeader>
                                  <CardContent className="space-y-3 text-sm text-slate-300">
                                    <div>
                                      <p className="text-xs text-slate-500">内容摘录</p>
                                      <p>{section.excerpt}</p>
                                    </div>
                                    <div>
                                      <p className="text-xs text-slate-500">结构摘要</p>
                                      <p>{section.summary}</p>
                                    </div>
                                    <div>
                                      <p className="text-xs text-slate-500">思路分析</p>
                                      <p>{section.thinking}</p>
                                    </div>
                                    <div className="flex flex-wrap gap-2">
                                      {section.keywords.map((item) => (
                                        <Badge key={item.term} className="bg-slate-800 text-slate-200">
                                          {item.term}
                                        </Badge>
                                      ))}
                                    </div>
                                  </CardContent>
                                </Card>
                              ))}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  <TabsContent value="keywords" className="space-y-6">
                    {result.documents.map((doc) => (
                      <Card key={doc.name} className="border-slate-800 bg-slate-950/40">
                        <CardHeader>
                          <CardTitle className="text-base">{doc.name} - 关键词画像</CardTitle>
                          <CardDescription>关键词频率与权重</CardDescription>
                        </CardHeader>
                        <CardContent>
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>关键词</TableHead>
                                <TableHead>权重</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {doc.keywords.map((item) => (
                                <TableRow key={item.term}>
                                  <TableCell className="font-medium text-slate-100">
                                    {item.term}
                                  </TableCell>
                                  <TableCell className="text-slate-300">
                                    {(item.score * 100).toFixed(2)}%
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </CardContent>
                      </Card>
                    ))}
                  </TabsContent>

                  <TabsContent value="charts" className="space-y-6">
                    <Card className="border-slate-800 bg-slate-950/40">
                      <CardHeader>
                        <CardTitle className="text-base">关键词热度图</CardTitle>
                        <CardDescription>高频关键词用于生成研究结论图。</CardDescription>
                      </CardHeader>
                      <CardContent className="h-72">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={result.keyword_chart}>
                            <XAxis dataKey="name" stroke="#94a3b8" />
                            <YAxis stroke="#94a3b8" />
                            <Tooltip
                              contentStyle={{
                                background: "#0f172a",
                                border: "1px solid #1e293b",
                                color: "#e2e8f0",
                              }}
                            />
                            <Bar dataKey="value" fill="#6366f1" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                    <Card className="border-slate-800 bg-slate-950/40">
                      <CardHeader>
                        <CardTitle className="text-base">研究结论图</CardTitle>
                        <CardDescription>展示核心关键词与共现关系。</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-3 text-sm text-slate-300">
                        <div className="flex flex-wrap gap-2">
                          {result.conclusion_graph.nodes.map((node) => (
                            <Badge key={node} className="bg-indigo-500/20 text-indigo-200">
                              {node}
                            </Badge>
                          ))}
                        </div>
                        <div className="space-y-2">
                          {result.conclusion_graph.edges.map((edge, index) => (
                            <div
                              key={`${edge.source}-${edge.target}-${index}`}
                              className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2"
                            >
                              <span>
                                {edge.source} → {edge.target}
                              </span>
                              <span className="text-xs text-slate-400">权重 {edge.weight}</span>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                     {result.document_type && result.enhanced_charts && (
                       <EnhancedCharts
                         enhancedCharts={result.enhanced_charts}
                         documentType={result.document_type}
                         recommendedCharts={result.recommended_charts}
                       />
                     )}
                  </TabsContent>

                  <TabsContent value="comparison" className="space-y-6">
                    <Card className="border-slate-800 bg-slate-950/40">
                      <CardHeader>
                        <CardTitle className="text-base">文献相似度矩阵</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>文档</TableHead>
                              {result.documents.map((doc) => (
                                <TableHead key={doc.name}>{doc.name}</TableHead>
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {result.documents.map((doc, rowIndex) => (
                              <TableRow key={doc.name}>
                                <TableCell className="font-medium text-slate-100">
                                  {doc.name}
                                </TableCell>
                                {result.comparison.similarity[rowIndex].map((value, index) => (
                                  <TableCell key={`${doc.name}-${index}`}> {value.toFixed(2)} </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </CardContent>
                    </Card>
                    <Card className="border-slate-800 bg-slate-950/40">
                      <CardHeader>
                        <CardTitle className="text-base">重叠关键词</CardTitle>
                      </CardHeader>
                      <CardContent className="flex flex-wrap gap-2">
                        {result.comparison.overlap_keywords.map((keyword) => (
                          <Badge key={keyword} className="bg-slate-800 text-slate-200">
                            {keyword}
                          </Badge>
                        ))}
                      </CardContent>
                    </Card>
                    <Card className="border-slate-800 bg-slate-950/40">
                      <CardHeader>
                        <CardTitle className="text-base">差异关键词</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm text-slate-300">
                        {Object.entries(result.comparison.unique_keywords).map(([doc, keywords]) => (
                          <div key={doc}>
                            <p className="font-medium text-slate-100">{doc}</p>
                            <p>{keywords.join("、") || "暂无"}</p>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="conflict" className="space-y-6">
                    <Card className="border-slate-800 bg-slate-950/40">
                      <CardHeader>
                        <CardTitle className="text-base">观点冲突检测</CardTitle>
                        <CardDescription>基于情感倾向和关键词共现判断。</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {result.conflicts.length === 0 && (
                          <p className="text-sm text-slate-400">未检测到明显冲突。</p>
                        )}
                        {result.conflicts.map((conflict, index) => (
                          <div
                            key={`${conflict.topic}-${index}`}
                            className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-3 text-sm text-rose-100"
                          >
                            <p className="font-medium">主题：{conflict.topic}</p>
                            <p className="mt-2">文献 A：{conflict.doc_a}</p>
                            <p className="mt-1">文献 B：{conflict.doc_b}</p>
                            <p className="mt-2 text-xs text-rose-200">
                              情绪差值：{Math.abs(conflict.sentiment_a - conflict.sentiment_b).toFixed(2)}
                            </p>
                          </div>
                        ))}
                      </CardContent>
                    </Card>
                  </TabsContent>
                </Tabs>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default App
