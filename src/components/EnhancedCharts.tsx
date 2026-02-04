import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from "recharts";


interface EnhancedChartsProps {
  enhancedCharts: {
    wordcloud: Array<{ text: string; value: number }>;
    pie: Array<{ name: string; value: number }>;
    timeline: Array<{ date: string; event: string; importance: number }>;
    entity_network: {
      nodes: Array<{ id: string; group: number }>;
      links: Array<{ source: string; target: string; value: number }>;
    };
  };
  documentType: string;
  recommendedCharts: string[];
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export default function EnhancedCharts({ enhancedCharts, documentType, recommendedCharts }: EnhancedChartsProps) {
  const chartNames: Record<string, string> = {
    "keyword_barchart": "关键词柱状图",
    "conclusion_graph": "结论关系图",
    "section_structure": "章节结构图",
    "reference_distribution": "参考文献分布",
    "timeline": "时间线图",
    "entity_relationship": "实体关系图",
    "code_distribution": "代码分布图",
    "api_statistics": "API统计图",
    "named_entity": "命名实体图",
    "sentiment_analysis": "情感分析图",
    "wordcloud": "关键词词云",
    "category_pie": "分类饼图"
  };

  return (
    <div className="space-y-6">
      <Card className="border-slate-800 bg-slate-950/40">
        <CardHeader>
          <CardTitle className="text-base">📊 个性化图表推荐</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-4">
            <Badge className="bg-indigo-500/20 text-indigo-200">文档类型: {documentType}</Badge>
            {recommendedCharts.map((chart, idx) => (
              <Badge key={idx} variant="outline" className="text-xs">
                {chartNames[chart] || chart}
              </Badge>
            ))}
          </div>
          <p className="text-sm text-slate-300">
            根据文档内容智能推荐{recommendedCharts.length}种图表，点击下方标签查看不同可视化效果。
          </p>
        </CardContent>
      </Card>

      <Tabs defaultValue="wordcloud" className="w-full">
        <TabsList className="grid grid-cols-4 mb-4">
          <TabsTrigger value="wordcloud">词云</TabsTrigger>
          <TabsTrigger value="pie">分类分布</TabsTrigger>
          <TabsTrigger value="timeline">时间线</TabsTrigger>
          <TabsTrigger value="network">实体网络</TabsTrigger>
        </TabsList>

        <TabsContent value="wordcloud">
          <Card className="border-slate-800 bg-slate-950/40">
            <CardHeader>
              <CardTitle className="text-base">关键词词云</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3 p-4 min-h-[200px] items-center justify-center">
                {enhancedCharts.wordcloud.map((item, idx) => {
                  const size = Math.max(14, Math.min(36, item.value / 50));
                  return (
                    <div
                      key={idx}
                      className="px-3 py-1 rounded-full bg-slate-800/50 border border-slate-700 hover:bg-slate-700/50 transition-all"
                      style={{
                        fontSize: `${size}px`,
                        fontWeight: size > 24 ? 'bold' : 'normal',
                        opacity: 0.7 + (size / 40) * 0.3
                      }}
                    >
                      {item.text}
                    </div>
                  );
                })}
              </div>
              <p className="text-xs text-slate-400 mt-4">字体大小表示关键词重要性</p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="pie">
          <Card className="border-slate-800 bg-slate-950/40">
            <CardHeader>
              <CardTitle className="text-base">分类分布饼图</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={enhancedCharts.pie}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={(entry) => `${entry.name}: ${entry.value}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {enhancedCharts.pie.map((_entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="timeline">
          <Card className="border-slate-800 bg-slate-950/40">
            <CardHeader>
              <CardTitle className="text-base">时间线图</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {enhancedCharts.timeline.map((item, idx) => (
                  <div key={idx} className="flex items-start gap-4">
                    <div className="flex flex-col items-center">
                      <div className="w-3 h-3 rounded-full bg-indigo-500 mt-1"></div>
                      {idx < enhancedCharts.timeline.length - 1 && (
                        <div className="w-0.5 h-full bg-slate-700 mt-1"></div>
                      )}
                    </div>
                    <div className="flex-1 pb-4">
                      <div className="flex justify-between">
                        <span className="font-medium text-slate-100">{item.date}</span>
                        <Badge className="bg-slate-800">重要性: {item.importance}</Badge>
                      </div>
                      <p className="text-sm text-slate-300 mt-1">{item.event}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="network">
          <Card className="border-slate-800 bg-slate-950/40">
            <CardHeader>
              <CardTitle className="text-base">实体关系网络</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  {enhancedCharts.entity_network.nodes.map((node, idx) => (
                    <Badge key={idx} className={`bg-slate-800 text-slate-200`}>
                      {node.id}
                    </Badge>
                  ))}
                </div>
                <div className="text-sm text-slate-300">
                  <p className="mb-2">实体关系连接：</p>
                  <div className="space-y-2">
                    {enhancedCharts.entity_network.links.map((link, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <span className="text-indigo-300">{link.source}</span>
                        <span className="text-slate-500">——({link.value})——</span>
                        <span className="text-indigo-300">{link.target}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}