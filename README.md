# AI 文档分析平台

基于React + TypeScript + Vite构建的AI文档分析平台，支持PDF、Word、TXT等多种格式文档上传，使用DeepSeek AI进行智能分析。

## 功能特性

- 📄 多格式文档支持（PDF、Word、TXT）
- 🤖 DeepSeek AI智能分析
- 📊 文档结构分析
- 🔑 关键词提取
- 📈 可视化图表展示
- 🚀 快速部署到Vercel

## 快速开始

### 本地开发

1. 克隆项目
2. 安装依赖：`npm install`
3. 启动开发服务器：`npm run dev`

### 部署到Vercel

#### 1. 设置API密钥

**重要：** 请先在Vercel中设置环境变量：

1. 登录 [Vercel控制台](https://vercel.com)
2. 进入项目设置
3. 点击 **Settings → Environment Variables**
4. 添加以下环境变量：
   - `OPENROUTER_API_KEY`: 你的OpenRouter API密钥（从 [OpenRouter](https://openrouter.ai/) 获取）
   - **注意：** 如果API密钥无效，AI分析功能将自动回退到传统算法，但体验会打折扣。请确保使用有效的API密钥。

#### 2. 部署项目

项目已配置Vercel，连接到GitHub仓库后会自动部署。

## 使用说明

1. 访问部署的网站
2. 上传文档（PDF、Word、TXT格式）
3. 等待AI分析完成
4. 查看分析结果：摘要、关键词、文档结构、可视化图表

## 安全注意事项

- **不要将API密钥上传到GitHub**
- API密钥应存储在环境变量中
- 生产环境使用环境变量配置

## 技术栈

- React + TypeScript
- Vite
- Tailwind CSS
- Python (后端API)
- DeepSeek AI (通过OpenRouter)
- Vercel (部署)

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
