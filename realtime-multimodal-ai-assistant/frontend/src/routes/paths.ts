// Base route constants
const BASE = {
  GBB: '/gbb-ai',
  ZEN: '/gbb-ai/zen',
  APPS: '/gbb-ai/apps',
};

// Path generator utility
const createPath = (base: string, suffix: string = ''): string => `${base}${suffix}`;
const createDetailPath =
  (base: string) =>
  (id: string): string =>
    `${base}/${id}`;
const createEditPath =
  (base: string) =>
  (id: string): string =>
    `${base}/${id}/edit`;

// Build route definitions
export const paths = {
  page404: '/404',

  // Documentation section
  documentation: {
    root: createPath(BASE.GBB, '/documentation'),
    introduction: createPath(BASE.GBB, '/documentation/introduction'),
    setupToUse: createPath(BASE.GBB, '/documentation/setup-to-use'),
    kmm: {
      create: createPath(BASE.GBB, '/documentation/create-kb'),
      chat: createPath(BASE.GBB, '/documentation/chat-with-kb'),
    },
    functions: {
      preview: createPath(BASE.GBB, '/documentation/functions-preview'),
    },
    gpts: {
      create: createPath(BASE.GBB, '/documentation/create-custom-gpt'),
      orchestrate: createPath(BASE.GBB, '/documentation/orchestrate-custom-gpts'),
    },
    applications: {
      aoaiWorkbench: createPath(BASE.GBB, '/documentation/aoai-workbench'),
      tvCopilot: createPath(BASE.GBB, '/documentation/tv-copilot'),
      chatDa: createPath(BASE.GBB, '/documentation/ai-data-analyzer'),
    },
    author: createPath(BASE.GBB, '/documentation/about-author'),
    faqs: createPath(BASE.GBB, '/documentation/faqs'),
    changeLog: createPath(BASE.GBB, '/documentation/changelog'),
  },

  // Main GBB AI routes
  gbbai: (() => {
    // Create base paths for sections
    const chatBase = `${BASE.GBB}/chat`;
    const imageBase = `${BASE.GBB}/image`;
    const tracingBase = `${BASE.GBB}/tracing`;
    const deepResearchBase = `${BASE.GBB}/deep-research`;
    const kbBase = `${BASE.GBB}/kb`;
    const functionBase = `${BASE.GBB}/function`;
    const agentsBase = `${BASE.GBB}/agents`;
    const appGalleryBase = `${BASE.GBB}/app-gallery`;
    const videoBase = `${appGalleryBase}/ai-video-analyzer`;

    return {
      root: BASE.GBB,

      tracing: {
        root: tracingBase,
        details: createDetailPath(tracingBase),
      },

      chat: {
        root: chatBase,
      },

      image: {
        root: imageBase,
      },

      deepResearch: {
        root: deepResearchBase,
        details: createDetailPath(deepResearchBase),
        edit: createEditPath(deepResearchBase),
        chat: createDetailPath(`${deepResearchBase}/chat`),
        directChat: `${deepResearchBase}/chat`,
      },

      kb: {
        root: kbBase,
        details: createDetailPath(kbBase),
        edit: createEditPath(kbBase),
        chunks: (kbId: string, fileId: string) => `${kbBase}/${kbId}/files/${fileId}/chunks`,
      },

      function: {
        root: functionBase,
        details: createDetailPath(functionBase),
        edit: createEditPath(functionBase),
      },

      agents: {
        root: agentsBase,
        create: `${agentsBase}/create`,
        details: createDetailPath(agentsBase),
        edit: (id: string) => `${agentsBase}/edit/${id}`,
        workbench: `${agentsBase}/workbench`,
      },

      agentFleet: `${BASE.GBB}/agent-fleet`,

      appGallery: {
        root: appGalleryBase,
        details: createDetailPath(appGalleryBase),
        edit: createEditPath(appGalleryBase),

        customGpt: {
          create: `${appGalleryBase}/custom-gpt/create`,
          edit: (id: string) => `${appGalleryBase}/custom-gpt/edit/${id}`,
        },

        aiVideo: {
          root: videoBase,
          list: videoBase,
          details: createDetailPath(videoBase),
          edit: (id: string) => `${videoBase}/${id}/edit`,
          startFromTime: (id: string, time: number) => `${videoBase}/${id}/${time}`,
        },

        smartDocParser: createDetailPath(appGalleryBase),
      },

      o1Usescases: {
        root: `${BASE.GBB}/o1-usecases`,
        details: (id: string) => `${BASE.GBB}/o1-usecases/${id}`,
      },

      workshop: {
        agent: { list: `${BASE.GBB}/workshop/agent/list` },
        rag: { list: `${BASE.GBB}/workshop/rag/list` },
      },

      user: {
        root: `${BASE.GBB}/user`,
        account: `${BASE.GBB}/user/account`,
      },
    };
  })(),

  // Single app routes
  singleApp: {
    app: (id: string) => `${BASE.APPS}/${id}`,
  },

  // Zen mode routes
  gbbai_zen: (() => {
    // Create base paths for zen sections
    const tracingBase = `${BASE.ZEN}/tracing`;
    const kbBase = `${BASE.ZEN}/kb`;
    const functionBase = `${BASE.ZEN}/function`;
    const agentsBase = `${BASE.ZEN}/agents`;
    const appGalleryBase = `${BASE.ZEN}/app-gallery`;
    const videoBase = `${appGalleryBase}/ai-video-analyzer`;

    return {
      root: BASE.ZEN,

      tracing: {
        root: tracingBase,
        details: createDetailPath(tracingBase),
      },

      kb: {
        root: kbBase,
        details: createDetailPath(kbBase),
        edit: createEditPath(kbBase),
      },

      function: {
        root: functionBase,
        details: createDetailPath(functionBase),
        edit: createEditPath(functionBase),
      },

      agents: {
        root: agentsBase,
        create: `${agentsBase}/create`,
        details: createDetailPath(agentsBase),
        edit: (id: string) => `${agentsBase}/edit/${id}`,
        workbench: `${agentsBase}/workbench`,
      },

      agentFleet: `${BASE.ZEN}/chat`,

      appGallery: {
        root: appGalleryBase,
        details: createDetailPath(appGalleryBase),
        edit: createEditPath(appGalleryBase),

        customGpt: {
          create: `${appGalleryBase}/custom-gpt/create`,
          edit: (id: string) => `${appGalleryBase}/custom-gpt/edit/${id}`,
        },

        aiVideo: {
          root: videoBase,
          list: videoBase,
          edit: (id: string) => `${videoBase}/${id}/edit`,
          details: createDetailPath(videoBase),
          startFromTime: (id: string, time: number) => `${videoBase}/${id}/${time}`,
        },
      },

      user: {
        root: `${BASE.ZEN}/user`,
        account: `${BASE.ZEN}/user/account`,
      },
    };
  })(),
};
