// --------------------------------------------------------------------------------------------------------------------
// <copyright file="FixClientService.cs" company="MBS Source">
//   Copyright © 2016 MBS Source. All Rights Reserved.
// </copyright>
// <summary>
//   Defines the FixClientService type.
// </summary>
// --------------------------------------------------------------------------------------------------------------------

namespace FixClientLite
{
    using System;
    using System.Configuration;
    using System.Runtime.InteropServices;
    using System.ServiceProcess;
    using System.Timers;

    /// <summary>The fix client service.</summary>
    #if WINDOWS
         public partial class FixClientService : ServiceBase
    #else
        public partial class FixClientService //: ServiceBase
    #endif
    {
        private static log4net.ILog logger = log4net.LogManager.GetLogger(System.Reflection.MethodBase.GetCurrentMethod().DeclaringType);
        public FixClient client;
        public FixClientService()
        {
            InitializeComponent();
            client = new FixClient();
        }

        #if WINDOWS
                 protected override void OnStart(string[] args)
                {
                    System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("en-US");
                    logger.Info("MBSSourceFIXProduction Service started!");
                }

                protected override void OnStop()
                {
                }
        #else
        
        #endif

        

        public void SelfHost(string[] args)
        {
        #if WINDOWS
                    if (Environment.UserInteractive 
                        )
                    {
                        Console.Title = "MBS FixClient";
                        logger.Info("MBSSource  started!");
                        //this.OnStart(args);
                        Console.WriteLine("Press [ENTER] to stop the service");
                        Console.Read();
                        //this.OnStop();
                    }
                    else
                    {
                        ServiceBase.Run(this);
                    }
        #else
                    Console.Title = "MBS FixClient";
                    logger.Info("MBSSource  started!");
                    Console.WriteLine("Press [ENTER] to stop the service");
                    Console.Read();
        #endif
        }
    }
}
